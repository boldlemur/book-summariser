# app.py — EPUB/PDF summarizer with OPF spine + NCX TOC + robust href/anchor resolution
# UX: One-card-per-chapter, Book Themes (not "roll-ups"), clean labels, no techy paths by default.

import os, io, time, tempfile, posixpath, re, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# EPUB/MOBI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from lxml import etree
from urllib.parse import urldefrag

# PDF
import pdfplumber

# ---------------- Config ----------------
DEFAULT_CHAT_MODEL  = st.secrets.get("OPENAI_CHAT_MODEL",  os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini"))
DEFAULT_EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

TARGET_SECTION_TOKENS = 2200  # internal chunk target (in tokens) — UI always merges to chapters
EXEC_SUMMARY_TOKENS   = 1200
THEME_SUMMARY_TOKENS  = 900
CHAPTER_MAP_TOKENS    = 700
GLOSSARY_TOKENS       = 800
CLAIMS_TOKENS         = 900
API_SLEEP             = 0.5

import streamlit as st

def check_password() -> bool:
    """Simple password gate using st.secrets + session_state."""
    def _password_entered():
        pw = st.session_state.get("password", "")
        if pw and pw == st.secrets.get("APP_PASSWORD"):
            st.session_state["password_ok"] = True
            # Don’t keep the plaintext password in memory
            del st.session_state["password"]
        else:
            st.session_state["password_ok"] = False

    if st.session_state.get("password_ok"):
        return True

    st.title("Login")
    st.text_input("Password", type="password", on_change=_password_entered, key="password")
    if st.session_state.get("password_ok") is False:
        st.error("Wrong password. Try again.")
    st.stop()


# ---------------- Utilities ----------------
def get_client():
    if OpenAI is None:
        st.error("Missing openai package. Add openai>=1.35.0 to requirements.")
        st.stop()
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("Set OPENAI_API_KEY in .streamlit/secrets.toml.")
        st.stop()
    return OpenAI(api_key=api_key)

def approx_tokens(s: str) -> int: return max(1, int(len(s)/4))
def batched(xs, n):
    b=[]
    for x in xs:
        b.append(x)
        if len(b)==n: yield b; b=[]
    if b: yield b

# ---------------- Data models ----------------
@dataclass
class Block:
    href: str
    anchor: Optional[str]
    kind: str
    text: str
    order: int

@dataclass
class Chunk:
    title: str
    level: int
    source: str
    range_hint: str
    blocks: List[Block]

# ---------------- EPUB helpers ----------------
CONTENT_TAGS = {"h1","h2","h3","h4","h5","h6","p","li","blockquote","figcaption"}

def _normalize_path(p: str) -> str:
    p = p.replace("\\","/")
    p = posixpath.normpath(p)
    return p

def _resolve_href(href_from_toc: str, all_doc_paths: List[str]) -> Optional[str]:
    cand = _normalize_path(href_from_toc)
    if "#" in cand: cand = cand.split("#",1)[0]
    lc = {p.lower(): p for p in all_doc_paths}
    if cand.lower() in lc: return lc[cand.lower()]
    parts = cand.split("/")
    for i in range(len(parts)):
        tail = "/".join(parts[i:])
        if tail.lower() in lc: return lc[tail.lower()]
    base = posixpath.basename(cand).lower()
    matches = [p for p in all_doc_paths if posixpath.basename(p).lower()==base]
    if len(matches)==1: return matches[0]
    matches = sorted([p for p in all_doc_paths if p.lower().endswith(base)], key=len)
    if matches: return matches[0]
    return None

def _parse_html_blocks_and_anchor_map(html: bytes, href: str) -> Tuple[List[Block], Dict[str, int]]:
    """
    Content blocks + anchor->block map per XHTML.
    If we can't find semantic tags, synthesize blocks from body text.
    """
    soup = BeautifulSoup(html, "lxml")

    pending_ids: List[str] = []
    blocks_local: List[Block] = []
    anchor_local_index: Dict[str, int] = {}
    local_order = 0
    last_content_idx: Optional[int] = None

    def register_block(text: str, tag_name: str, el_has_id: Optional[str] = None):
        nonlocal local_order, last_content_idx
        if not text.strip(): return
        anchor = f"#{el_has_id}" if el_has_id else None
        blocks_local.append(Block(href=href, anchor=anchor, kind=tag_name, text=" ".join(text.split()), order=local_order))
        for pid in pending_ids:
            anchor_local_index[pid] = len(blocks_local) - 1
        pending_ids.clear()
        last_content_idx = len(blocks_local) - 1
        local_order += 1

    # primary: semantic tags
    for el in soup.descendants:
        name = getattr(el, "name", None)
        if not name: continue
        if hasattr(el, "has_attr") and el.has_attr("id"): pending_ids.append(el.get("id"))
        if name in {"nav","script","style"}: continue
        if name in CONTENT_TAGS:
            txt = el.get_text(" ", strip=True)
            if txt:
                el_id = el.get("id") if hasattr(el, "has_attr") and el.has_attr("id") else None
                register_block(txt, name, el_id)

    # fallback: synthesize
    if not blocks_local:
        body = soup.body or soup
        raw = body.get_text("\n", strip=True)
        raw = "\n".join([ln.strip() for ln in raw.splitlines() if ln.strip()])
        if raw:
            buf, acc = [], 0
            for sent in re.split(r"(?<=[\.\!\?])\s+", raw):
                if not sent.strip(): continue
                buf.append(sent.strip()); acc += len(sent)
                if acc >= 600:
                    register_block(" ".join(buf), "synthetic")
                    buf, acc = [], 0
            if buf: register_block(" ".join(buf), "synthetic")
            if pending_ids and last_content_idx is not None:
                for pid in pending_ids: anchor_local_index[pid] = last_content_idx
                pending_ids.clear()

    return blocks_local, anchor_local_index

def _get_toc_entries(book) -> List[Tuple[str, Optional[str], int, str]]:
    """
    TOC as [(href_raw, frag, level, title)].
    Detects NCX by mime or suffix; NAV by presence of <nav epub:type="toc">.
    """
    out: List[Tuple[str, Optional[str], int, str]] = []

    # NCX (EPUB2)
    ncx_items = []
    for it in book.get_items():
        mt = getattr(it, "media_type", "") or ""
        fn = (getattr(it, "file_name", "") or "").lower()
        if mt == "application/x-dtbncx+xml" or fn.endswith(".ncx"):
            ncx_items.append(it)
    for item in ncx_items:
        try:
            tree = etree.fromstring(item.get_content())
            ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
            def walk(el, level=1):
                for np in el.findall("ncx:navPoint", ns):
                    label = np.find("ncx:navLabel/ncx:text", ns)
                    content = np.find("ncx:content", ns)
                    title = (label.text or "").strip() if label is not None else ""
                    src = (content.get("src") or "").strip() if content is not None else ""
                    href_raw, frag = urldefrag(src)
                    out.append((href_raw, frag if frag else None, level, title))
                    walk(np, level + 1)
            navmap = tree.find("ncx:navMap", ns)
            if navmap is not None: walk(navmap, 1)
        except Exception:
            pass
    if out: return out

    # NAV (EPUB3)
    nav_like = []
    for it in book.get_items():
        mt = getattr(it, "media_type", "") or ""
        if mt in ("application/xhtml+xml","text/html"):
            try:
                soup = BeautifulSoup(it.get_content(), "lxml")
                if soup.find("nav", attrs={"epub:type":"toc"}) or soup.find("nav", {"role":"doc-toc"}):
                    nav_like.append(it)
            except Exception:
                continue
    for item in nav_like:
        try:
            soup = BeautifulSoup(item.get_content(), "lxml")
            nav = (soup.find("nav", attrs={"epub:type":"toc"})
                   or soup.find("nav", {"role":"doc-toc"})
                   or soup.find("nav"))
            if not nav: continue
            def walk_list(ul, level=1):
                for li in ul.find_all("li", recursive=False):
                    a = li.find("a", href=True)
                    if a:
                        href_raw, frag = urldefrag(a["href"])
                        title = a.get_text(" ", strip=True)
                        out.append((href_raw, frag if frag else None, level, title))
                    sub = li.find("ol") or li.find("ul")
                    if sub: walk_list(sub, level+1)
            for lst in nav.find_all(["ol","ul"], recursive=False): walk_list(lst,1)
        except Exception:
            pass
    if out: return out

    # Fallback: ebooklib toc prop/method
    toc_obj = getattr(book, "toc", None)
    if toc_obj:
        def flat(node, level=1):
            if isinstance(node, list):
                for n in node: flat(n, level)
            elif isinstance(node, epub.Section):
                for ch in node.subitems: flat(ch, level+1)
            elif isinstance(node, epub.Link):
                href_raw, frag = urldefrag(node.href or "")
                out.append((href_raw, frag if frag else None, level, node.title or ""))
        try: flat(toc_obj,1)
        except Exception: pass
        if out: return out

    get_toc = getattr(book, "get_toc", None)
    if callable(get_toc):
        try:
            obj = get_toc()
            out=[]
            def flat2(node, level=1):
                if isinstance(node, list):
                    for n in node: flat2(n, level)
                elif isinstance(node, epub.Section):
                    for ch in node.subitems: flat2(ch, level+1)
                elif isinstance(node, epub.Link):
                    href_raw, frag = urldefrag(node.href or "")
                    out.append((href_raw, frag if frag else None, level, node.title or ""))
            flat2(obj,1)
            if out: return out
        except Exception:
            pass
    return []

def read_epub_from_bytes(data: bytes):
    """Return (meta, blocks, toc_points (with global idx), spine_hrefs)."""
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(data); path = tmp.name
    book = epub.read_epub(path)

    meta = {
        "title":  (book.get_metadata('DC','title')[0][0] if book.get_metadata('DC','title') else "Untitled"),
        "author": ", ".join(m[0] for m in book.get_metadata('DC','creator')) if book.get_metadata('DC','creator') else "Unknown",
        "lang":   (book.get_metadata('DC','language')[0][0] if book.get_metadata('DC','language') else "und"),
    }

    # Spine order
    spine_ids = [sid if isinstance(sid, str) else sid[0] for sid in getattr(book,"spine",[])]
    spine_items = []
    for sid in spine_ids:
        it = book.get_item_with_id(sid)
        if it and it.get_type()==ebooklib.ITEM_DOCUMENT:
            spine_items.append(it)
    if not spine_items:
        spine_items = sorted([it for it in book.get_items() if it.get_type()==ebooklib.ITEM_DOCUMENT],
                             key=lambda it: _normalize_path(it.file_name))

    # Parse documents in spine order; build blocks + robust anchor map
    blocks: List[Block] = []
    href_first_idx: Dict[str,int] = {}
    anchor_to_global: Dict[Tuple[str,str], int] = {}
    for it in spine_items:
        href = _normalize_path(it.file_name)
        html = it.get_content()
        local_blocks, anchor_local = _parse_html_blocks_and_anchor_map(html, href)
        base = len(blocks)
        for i, b in enumerate(local_blocks):
            blocks.append(Block(href=b.href, anchor=b.anchor, kind=b.kind, text=b.text, order=base+i))
        if local_blocks:
            href_first_idx.setdefault(href, base)
        for frag, local_idx in anchor_local.items():
            anchor_to_global[(href, frag)] = base + local_idx

    # Build TOC entries
    all_doc_paths = [b for b in href_first_idx.keys()]
    toc_raw = _get_toc_entries(book)

    toc_points: List[Dict[str,str]] = []
    last_idx = -1
    for href_raw, frag, level, title in toc_raw:
        resolved = _resolve_href(href_raw, all_doc_paths)
        if not resolved: continue
        idx = None
        if frag and (resolved, frag) in anchor_to_global:
            idx = anchor_to_global[(resolved, frag)]
        elif resolved in href_first_idx:
            idx = href_first_idx[resolved]
        if idx is None and resolved in href_first_idx:
            start = href_first_idx[resolved]
            for j in range(start, min(start+200, len(blocks))):
                if blocks[j].href != resolved: break
                if title and title.strip() and title.lower() in blocks[j].text.lower():
                    idx = j; break
        if idx is None or idx <= last_idx: continue
        toc_points.append({"href": resolved, "frag": (frag or ""), "level": str(level or 1), "title": title or "Section", "idx": str(idx)})
        last_idx = idx

    spine_hrefs = [_normalize_path(it.file_name) for it in spine_items]
    return meta, blocks, toc_points, spine_hrefs

def chunks_from_epub(meta, blocks, toc_points, spine_hrefs, target_tokens) -> List[Chunk]:
    # Prefer TOC-based chunks
    chunks: List[Chunk] = []
    if toc_points:
        pts = [{"idx": int(p["idx"]), "title": p["title"], "level": int(p["level"]), "href": p["href"]} for p in toc_points]
        pts.sort(key=lambda x: x["idx"])
        for i, p in enumerate(pts):
            a = p["idx"]; b = pts[i + 1]["idx"] if i + 1 < len(pts) else len(blocks)
            blks = blocks[a:b]
            if not blks: continue
            start, end = blks[0], blks[-1]
            rng = f"{p['title']} ({start.href} … {end.href})"
            chunks.append(Chunk(title=p["title"], level=min(p["level"], 3), source="epub", range_hint=rng, blocks=blks))

    # safeguard: if TOC produced too few chunks, fall back to per-file chunks
    if not chunks or len(chunks) < max(1, int(0.4 * len(spine_hrefs))):
        file_to_blocks: Dict[str, List[Block]] = {}
        for b in blocks:
            file_to_blocks.setdefault(b.href, []).append(b)
        chunks = []
        for href in spine_hrefs:
            blks = file_to_blocks.get(href, [])
            if not blks: continue
            start,end = blks[0], blks[-1]
            heading = next((b.text for b in blks if b.kind in {"h1","h2","h3"}), None)
            title = heading if heading else os.path.splitext(os.path.basename(href))[0].replace("_"," ").upper()
            rng = f"{start.href} … {end.href}"
            chunks.append(Chunk(title=title, level=2, source="epub", range_hint=rng, blocks=blks))

    # internal splitting for giant chapters (UI will merge later)
    final: List[Chunk] = []
    for ch in chunks:
        toks = sum(approx_tokens(x.text) for x in ch.blocks)
        if toks <= target_tokens or len(ch.blocks) < 8:
            final.append(ch)
        else:
            mid = len(ch.blocks)//2
            final.append(Chunk(ch.title+" (Part 1)", ch.level, ch.source, ch.range_hint, ch.blocks[:mid]))
            final.append(Chunk(ch.title+" (Part 2)", ch.level, ch.source, ch.range_hint, ch.blocks[mid:]))
    return final

# ---------- PDF helpers (drop-in replacement) ----------

PDF_FRONTMATTER_HINTS = [
    "©", "All rights reserved", "Limit of Liability/Disclaimer", "ISBN", "Library of Congress",
    "Printed in", "John Wiley", "Publisher", "permissions", "This edition first published"
]

def _looks_like_frontmatter(page_text: str) -> bool:
    t = (page_text or "").strip()
    if not t:
        return True
    hits = sum(1 for h in PDF_FRONTMATTER_HINTS if h.lower() in t.lower())
    # short/legal-ish or mostly headings with little prose
    shortish = len(t) < 1200
    return hits >= 2 or ("contents" in t.lower() and "chapter" not in t.lower() and shortish)

def join_paragraphs(page_text: str) -> List[str]:
    lines=[ln.rstrip() for ln in (page_text or "").splitlines()]
    paras,buf=[],[]
    for ln in lines:
        if not ln.strip():
            if buf: paras.append(" ".join(buf).strip()); buf=[]
        else:
            # glue hyphenated line breaks
            if ln.endswith("-") and not ln.endswith("--"): buf.append(ln[:-1])
            else: buf.append(ln)
    if buf: paras.append(" ".join(buf).strip())
    # compress stray multi-space splits unless they look like tabular TOC
    final=[]
    for p in paras:
        chunks=[c.strip() for c in re.split(r"\s{2,}", p) if c.strip()]
        final.extend(chunks if (len(chunks)<6) else [p])
    return [p for p in final if p]

def chunk_pdf_extract(pdf_bytes):
    """Return (blocks, paragraphs, page_texts). 1 block per paragraph, retains page ids."""
    blocks=[]; order=0; page_texts=[]
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i,page in enumerate(pdf.pages, start=1):
            try:
                text=page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                text=""
            page_texts.append(text)
            paras = join_paragraphs(text)
            for p in paras:
                if p.strip():
                    blocks.append(Block(href=f"page-{i}", anchor=None, kind="pagepara", text=p.strip(), order=order)); order+=1
    paragraphs=[b.text for b in blocks]
    return blocks, paragraphs, page_texts

# ----------- TOC & chapter-anchor detection -----------

_CHAPTER_RES = [
    re.compile(r"^\s*CHAPTER\s+(\d+)\b", re.I),
    re.compile(r"^\s*C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\b", re.I),  # spaced variant
]
def _is_all_caps_heading(s: str) -> bool:
    s2=s.strip()
    if not (5 <= len(s2) <= 90): return False
    letters=[c for c in s2 if c.isalpha()]
    if not letters: return False
    # allow punctuation/numbers but mostly caps
    return sum(1 for c in letters if c.isupper())/len(letters) > 0.8 and s2.endswith((".",":")) is False

def _parse_contents_map(page_texts: List[str]) -> List[Tuple[int,str]]:
    """
    Try to parse a 'Contents' page listing 'n  Title ..... page'.
    Returns list of (start_page_index_1based, title). Sorted by page.
    """
    candidates=[]
    for pi, txt in enumerate(page_texts, start=1):
        if "contents" in (txt or "").lower():
            # grab lines that look like "... , 35" or "... 35"
            for ln in (txt or "").splitlines():
                m = re.search(r"(.+?)\s+(\d{1,4})\s*$", ln.strip())
                if m:
                    title=m.group(1).strip().replace("  "," ")
                    try:
                        pg=int(m.group(2))
                        # filter obvious junk
                        if 1 <= pg <= len(page_texts) + 5 and len(title) > 3:
                            candidates.append((pg, title))
                    except:
                        pass
    # de-dup by page, keep earliest mention
    seen=set(); out=[]
    for pg,title in sorted(candidates, key=lambda x:(x[0], x[1])):
        if pg not in seen:
            seen.add(pg); out.append((pg,title))
    # filter if too few chapters detected
    return out if len(out) >= 5 else []

def _detect_chapter_anchors_from_body(page_texts: List[str]) -> List[Tuple[int,str]]:
    """
    Scan body text for clear chapter starts: CHAPTER n [+ next line as title] or strong ALL CAPS lines.
    Return list of (page_num, title).
    """
    anchors=[]
    for pi, txt in enumerate(page_texts, start=1):
        lines=[ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
        for li, ln in enumerate(lines):
            # CHAPTER n
            for rx in _CHAPTER_RES:
                m=rx.match(ln)
                if m:
                    # title might be on same line after number OR next non-empty line
                    title = ln
                    # prefer next line if it's short and not boilerplate
                    if li+1 < len(lines):
                        nxt = lines[li+1]
                        if len(nxt)<=120 and not any(tag in nxt.lower() for tag in ["copyright","publisher","isbn"]):
                            title = re.sub(r"^\s*(C\s*H\s*A\s*P\s*T\s*E\s*R|CHAPTER)\s+\d+\s*[:\-–]?\s*", "", nxt, flags=re.I)
                            title = title if title else f"Chapter {m.group(1)}"
                        else:
                            title = f"Chapter {m.group(1)}"
                    else:
                        title = f"Chapter {m.group(1)}"
                    anchors.append((pi, title.strip()))
                    break
            else:
                # strong ALL CAPS short heading near top of page, likely a chapter title
                if li <= 3 and _is_all_caps_heading(ln) and not ln.lower().startswith("contents"):
                    anchors.append((pi, ln.title()))
            if anchors and anchors[-1][0]==pi:
                break  # only one anchor per page
    # de-dup by page
    seen=set(); uniq=[]
    for pg,title in anchors:
        if pg not in seen:
            seen.add(pg); uniq.append((pg,title))
    return uniq

def _build_pdf_chapters_from_anchors(blocks: List[Block], anchors: List[Tuple[int,str]]) -> List[Chunk]:
    if not anchors: return []
    anchors_sorted = sorted(anchors, key=lambda x:x[0])
    # convert pages to block indices
    page_first_idx: Dict[int,int] = {}
    for i,b in enumerate(blocks):
        try:
            pg = int(b.href.split("-")[-1])
            page_first_idx.setdefault(pg, i)
        except:
            continue
    chapters=[]
    for i,(start_pg, title) in enumerate(anchors_sorted):
        a = page_first_idx.get(start_pg, None)
        if a is None: continue
        end_pg = anchors_sorted[i+1][0]-1 if i+1 < len(anchors_sorted) else None
        # find end idx by last block whose page <= end_pg
        if end_pg is None:
            b = len(blocks)
        else:
            # find first index of the page after end_pg (exclusive)
            next_idx = None
            for pg in range(end_pg+1, end_pg+2):  # page after end range
                next_idx = page_first_idx.get(pg, None)
                if next_idx is not None: break
            b = next_idx if next_idx is not None else len(blocks)
        blks = blocks[a:b]
        if not blks: continue
        rng = f"pp. {start_pg}–{int(blks[-1].href.split('-')[-1])}"
        chapters.append(Chunk(title=title, level=2, source="pdf", range_hint=rng, blocks=blks))
    return chapters

# ----------- Fallback semantic chunking -----------
def detect_candidate_headings(paragraphs: List[str]) -> List[int]:
    idxs=[]
    for i,s in enumerate(paragraphs):
        s2=s.strip()
        if len(s2)<=80:
            is_title_case = s2==s2.title()
            is_all_caps = (s2.upper()==s2) and any(c.isalpha() for c in s2)
            if (is_title_case or is_all_caps) and not s2.endswith(".") and s2.count(",")<=1 and s2.count(";")==0:
                idxs.append(i)
    return idxs

def embed_texts(client, texts, model):
    embs=[]
    for batch in batched(texts, 64):
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data: embs.append(d.embedding)
        time.sleep(API_SLEEP)
    return np.array(embs, dtype=np.float32)

def cosine_sim(a,b):
    a/= (np.linalg.norm(a, axis=1, keepdims=True)+1e-8)
    b/= (np.linalg.norm(b, axis=1, keepdims=True)+1e-8)
    return np.dot(a,b.T)

def texttiling_boundaries(paragraphs, client, embed_model):
    if len(paragraphs)<8: return [0,len(paragraphs)]
    embs = embed_texts(client, paragraphs, embed_model)
    sims=[float(cosine_sim(embs[i:i+1], embs[i-1:i]).squeeze()) for i in range(1,len(paragraphs))]
    arr=np.array(sims); mu=float(np.mean(arr)); sd=float(np.std(arr)+1e-6)
    z=(arr-mu)/sd; bounds=[0]
    for i,zi in enumerate(z, start=1):
        if zi<-1.0: bounds.append(i)
    if bounds[-1]!=len(paragraphs): bounds.append(len(paragraphs))
    return sorted(set(bounds))

def sections_from_pdf(blocks, paragraphs, page_texts, client, embed_model, target_tokens):
    """
    Prefer: TOC → anchors; else semantic fallback.
    Returns List[Chunk] with human-friendly chapter titles.
    """
    # 0) Damp frontmatter paragraphs from driving boundaries
    if any(_looks_like_frontmatter(t) for t in page_texts[:4]):
        # drop the first N pages if they are obviously frontmatter (keep blocks mapping)
        front_pages = set(i+1 for i,t in enumerate(page_texts[:6]) if _looks_like_frontmatter(t))
    else:
        front_pages = set()

    # 1) TOC-based chapters
    toc_anchors = _parse_contents_map(page_texts)
    if toc_anchors:
        chapters = _build_pdf_chapters_from_anchors(blocks, toc_anchors)
        if len(chapters) >= 5:
            return _split_oversized_pdf_chapters(chapters, target_tokens)

    # 2) Body anchor detection (CHAPTER n, etc.)
    body_anchors = _detect_chapter_anchors_from_body(page_texts)
    if body_anchors:
        chapters = _build_pdf_chapters_from_anchors(blocks, body_anchors)
        if len(chapters) >= 4:
            return _split_oversized_pdf_chapters(chapters, target_tokens)

    # 3) Semantic fallback (TextTiling over paragraphs)
    if not paragraphs: return []
    heading_idxs=set(detect_candidate_headings(paragraphs))
    bounds=texttiling_boundaries(paragraphs, client, embed_model)
    chunks=[]
    idx_to_block={i:b for i,b in enumerate(blocks)}
    chap_no=0
    for si in range(len(bounds)-1):
        a,b=bounds[si],bounds[si+1]
        if a==b: continue
        blks=[idx_to_block[i] for i in range(a,b)]
        # try to skip leading frontmatter-only spans
        if blks and int(blks[0].href.split("-")[-1]) in front_pages and chap_no==0:
            continue
        title = paragraphs[a] if a in heading_idxs else f"Chapter {chap_no+1}"
        start,end=blks[0], blks[-1]
        rng=f"pp. {start.href.split('-')[-1]}–{end.href.split('-')[-1]}"
        chunks.append(Chunk(title=title, level=2, source="pdf", range_hint=rng, blocks=blks))
        chap_no += 1

    return _split_oversized_pdf_chapters(chunks, target_tokens)

def _split_oversized_pdf_chapters(chunks: List[Chunk], target_tokens: int) -> List[Chunk]:
    final=[]
    for ch in chunks:
        toks=sum(approx_tokens(x.text) for x in ch.blocks)
        if toks<=target_tokens or len(ch.blocks)<8:
            final.append(ch)
        else:
            mid=len(ch.blocks)//2
            final.append(Chunk(ch.title+" (Part 1)", ch.level, ch.source, ch.range_hint, ch.blocks[:mid]))
            final.append(Chunk(ch.title+" (Part 2)", ch.level, ch.source, ch.range_hint, ch.blocks[mid:]))
    return final


# ---------------- LLM helpers ----------------
def llm_chat(client, system_prompt, user_prompt, model, temperature, max_tokens):
    last=None
    for _ in range(3):
        try:
            resp=client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last=e; time.sleep(API_SLEEP*2)
    raise last

def map_summarize_chunk(client, ch, chat_model):
    text="\n\n".join(b.text for b in ch.blocks)
    cite=ch.range_hint  # used for internal traceability; hidden by default
    sys=("You are a precise research assistant. Produce 6–12 EXTRACTIVE bullets from the text. "
         "Then write a 3–5 sentence mini-summary. Do not invent facts. No 'what's not in the text'.")
    usr=f"""Section: {ch.title}

Bullets (extractive, concise):
- 

Short summary (3–5 sentences):

TEXT:
{text}
"""
    content=llm_chat(client, sys, usr, chat_model, 0.1, CHAPTER_MAP_TOKENS)
    return {"title": ch.title, "source_range": cite, "notes_and_summary": content}

# ----- NEW: merge internal parts into proper chapters -----
def normalize_chapter_title(t:str)->str:
    return re.sub(r"\s*\(Part\s+\d+\)\s*$", "", t).strip()

def merge_internal_parts_to_chapters(chunks: List[Chunk]) -> List[Chunk]:
    groups: Dict[str, List[Chunk]] = {}
    for ch in chunks:
        key = normalize_chapter_title(ch.title)
        groups.setdefault(key, []).append(ch)
    merged: List[Chunk] = []
    for key, parts in groups.items():
        parts_sorted = sorted(parts, key=lambda c: c.title)  # Part 1, Part 2 order
        blocks = []
        for p in parts_sorted: blocks.extend(p.blocks)
        rng = f"{parts_sorted[0].blocks[0].href} … {parts_sorted[-1].blocks[-1].href}" if parts_sorted and parts_sorted[0].blocks and parts_sorted[-1].blocks else ""
        merged.append(Chunk(title=key, level=min(p.level for p in parts_sorted), source=parts_sorted[0].source, range_hint=rng, blocks=blocks))
    # Preserve book order by first occurrence
    first_pos = {}
    pos=0
    for ch in chunks:
        key = normalize_chapter_title(ch.title)
        if key not in first_pos:
            first_pos[key]=pos
            pos+=1
    merged.sort(key=lambda c: first_pos.get(c.title, 1e9))
    return merged

# ----- NEW: build Book Themes -----
def kmeans(vectors: np.ndarray, k: int, iters: int = 15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if k >= n:  # trivial
        return np.arange(n), vectors.copy()
    # init++ lite: pick one random, then farthest
    centers = [vectors[rng.randint(0, n)]]
    for _ in range(1, k):
        d2 = np.min(np.linalg.norm(vectors[:,None,:]-np.array(centers)[None,:,:], axis=2)**2, axis=1)
        probs = d2 / (d2.sum()+1e-9)
        idx = rng.choice(n, p=probs)
        centers.append(vectors[idx])
    centers = np.array(centers)
    for _ in range(iters):
        dists = np.linalg.norm(vectors[:,None,:]-centers[None,:,:], axis=2)
        labels = np.argmin(dists, axis=1)
        for j in range(k):
            pts = vectors[labels==j]
            if len(pts)>0: centers[j] = pts.mean(axis=0)
    dists = np.linalg.norm(vectors[:,None,:]-centers[None,:,:], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels, centers

def build_themes(client, chapters: List[Dict[str,Any]], embed_model: str, chat_model: str) -> List[Dict[str,Any]]:
    titles = [c["title"] for c in chapters]
    # Embed chapter titles (fast + good enough); if many, embed summaries instead
    texts = [f"{i+1}. {t}" for i,t in enumerate(titles)]
    embs = embed_texts(client, texts, embed_model)
    n = len(chapters)
    k = min(6, max(3, int(round(math.sqrt(n)))) )  # 3–6 themes
    labels, _ = kmeans(embs, k)
    groups: Dict[int, List[int]] = {}
    for i,l in enumerate(labels): groups.setdefault(int(l), []).append(i)

    themes=[]
    for gi, idxs in groups.items():
        idxs_sorted = sorted(idxs)
        ch_subset = [chapters[i] for i in idxs_sorted]
        # Name + summary with the model
        joined = "\n\n---\n\n".join([f"Chapter: {c['title']}\n\n{c['notes_and_summary']}" for c in ch_subset])
        sys = ("Summarize the common thread across these chapters. "
               "First provide a short THEME TITLE (3–6 words), then 1–2 paragraphs explaining the theme. "
               "Do not add facts not present in the material. Avoid citations.")
        usr = f"Chapters covered:\n" + "\n".join([f"- {c['title']}" for c in ch_subset]) + f"\n\nMaterial:\n{joined}"
        theme_text = llm_chat(client, sys, usr, chat_model, 0.3, THEME_SUMMARY_TOKENS)
        # Split a line as title if present; else fallback
        first_line, _, rest = theme_text.partition("\n")
        title = first_line.strip().strip("#-•: ")[:80] if len(first_line.strip())>=3 else "Theme"
        body  = rest.strip() if rest.strip() else theme_text.strip()
        themes.append({
            "title": title,
            "summary": body,
            "chapter_ids": idxs_sorted,
            "chapter_titles": [chapters[i]["title"] for i in idxs_sorted]
        })
    # Sort themes by earliest chapter index they include
    themes.sort(key=lambda t: min(t["chapter_ids"]) if t["chapter_ids"] else 1e9)
    return themes

# ----- Book-wide glossary & claims -----
def build_glossary(client, chapters: List[Dict[str,Any]], chat_model: str) -> str:
    sys = "From the chapter notes below, extract a compact book-wide glossary of key terms/names (20–60 entries). No inventions. No citations."
    joined = "\n\n".join([f"{c['title']}\n{c['notes_and_summary']}" for c in chapters])
    usr = joined
    return llm_chat(client, sys, usr, chat_model, 0.1, GLOSSARY_TOKENS)

def build_claims_index_by_theme(client, themes: List[Dict[str,Any]], chapters: List[Dict[str,Any]], chat_model: str) -> List[Dict[str,str]]:
    out=[]
    for th in themes:
        joined = "\n\n---\n\n".join([f"{chapters[i]['title']}\n{chapters[i]['notes_and_summary']}" for i in th["chapter_ids"]])
        sys = "Create a concise claims index (bulleted) for this theme using only the material provided. Reference chapters by title in parentheses. No inventions."
        usr = joined
        text = llm_chat(client, sys, usr, chat_model, 0.2, CLAIMS_TOKENS)
        out.append({"theme_title": th["title"], "claims": text})
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="📘 Book Summarizer (EPUB/PDF)", page_icon="📘", layout="wide")
st.title("📘 Book Summarizer")
st.caption("Chapters detected from EPUB TOC (OPF/NCX) or PDF sections. Clean chapter summaries, book themes, glossary, and claims index.")
check_password()

with st.sidebar:
    st.header("Settings")
    chat_model  = st.text_input("Chat Model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding Model", value=DEFAULT_EMBED_MODEL)
    TARGET_SECTION_TOKENS = st.number_input("Max tokens per internal chunk", 500, 6000, TARGET_SECTION_TOKENS, step=100, help="Internal safety chunking; UI merges to chapters.")
    API_SLEEP = st.number_input("API sleep (seconds)", 0.0, 3.0, API_SLEEP, step=0.1)
    st.divider()
    debug_show_ranges = st.toggle("Advanced: show debug ranges & paths", value=False)

uploaded = st.file_uploader("Upload a book/document", type=["epub","pdf"])

if uploaded:
    client = get_client()
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".epub"):
        st.info("Parsing EPUB (OPF spine + TOC)…")
        meta, blocks, toc_points, spine_hrefs = read_epub_from_bytes(data)
        chunks_internal = chunks_from_epub(meta, blocks, toc_points, spine_hrefs, TARGET_SECTION_TOKENS)

    elif name.endswith(".pdf"):
        st.info("Parsing PDF…")
        blocks, paragraphs, page_texts = chunk_pdf_extract(data)   # <- returns 3 now
        with st.spinner("Detecting chapters/sections…"):
            chunks_internal = sections_from_pdf(
                blocks, paragraphs, page_texts, client, embed_model, TARGET_SECTION_TOKENS
            )

    else:
        st.error("Unsupported file type.")
        st.stop()

    if not chunks_internal:
        st.error("No content blocks detected. If this is scanned/DRM-protected, try a text-based EPUB/PDF.")
        st.stop()


    # Merge (Part 1/2) into chapters for UI
    chapters_chunks = merge_internal_parts_to_chapters(chunks_internal)

    # ----- Table of chapters -----
    st.subheader("Chapters")
    rows=[]
    for i, ch in enumerate(chapters_chunks, start=1):
        est_tokens=sum(approx_tokens(b.text) for b in ch.blocks)
        row = {"#": i, "Chapter": ch.title[:160], "Blocks (internal)": len(ch.blocks), "≈Tokens (internal)": est_tokens}
        if debug_show_ranges: row["Debug Range"] = ch.range_hint
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ----- Per-chapter notes -----
    st.subheader("Chapter Summaries")
    per_chapter=[]
    prog=st.progress(0.0, text="Summarizing chapters…")
    for i, ch in enumerate(chapters_chunks):
        out=map_summarize_chunk(client, ch, chat_model)
        out["chapter_index"]=i
        per_chapter.append(out)
        prog.progress((i+1)/len(chapters_chunks), text=f"Summarized {i+1}/{len(chapters_chunks)}")
    st.success("Chapter summaries complete.")

    # ----- Book themes -----
    st.subheader("Book Themes")
    themes = build_themes(client, per_chapter, embed_model, chat_model)
    for t in themes:
        with st.expander(f"{t['title']}  — covers: " + ", ".join([f'Ch {i+1}' for i in t['chapter_ids']]), expanded=False):
            st.write(t["summary"])
            st.caption("Chapters: " + ", ".join(t["chapter_titles"]))

    # ----- Executive summary -----
    joined_all = "\n\n---\n\n".join([f"{c['title']}\n{c['notes_and_summary']}" for c in per_chapter])
    sys_exec = "Write a clear, book-wide executive summary (3–6 short paragraphs). No citations. No 'what’s not in the text'."
    exec_summary = llm_chat(client, sys_exec, joined_all, chat_model, 0.3, EXEC_SUMMARY_TOKENS)
    st.markdown("### 🧠 Executive Summary"); st.write(exec_summary)

    # ----- Glossary & Claims (book-wide) -----
    st.subheader("Glossary & Claims")
    glossary_text = build_glossary(client, per_chapter, chat_model)
    st.markdown("#### 📑 Book Glossary"); st.write(glossary_text)

    claims_by_theme = build_claims_index_by_theme(client, themes, per_chapter, chat_model)
    st.markdown("#### 📌 Claims Index (by Theme)")
    for c in claims_by_theme:
        with st.expander(c["theme_title"], expanded=False):
            st.write(c["claims"])

    # ----- Chapter cards -----
    st.subheader("Chapter Details")
    for i, sec in enumerate(per_chapter, start=1):
        header = f"Chapter {i}: {sec['title']}"
        if debug_show_ranges: header += f"  —  [{sec['source_range']}]"
        with st.expander(header, expanded=False):
            st.write(sec["notes_and_summary"])

    # ----- Exports -----
    st.subheader("Export")
    import json
    export = {
        "meta": meta if name.endswith(".epub") else {"title": uploaded.name, "author": "", "lang": ""},
        "chapters_table": rows,
        "chapters": per_chapter,
        "themes": themes,
        "executive_summary": exec_summary,
        "glossary": glossary_text,
        "claims_by_theme": claims_by_theme,
        "advanced_includes_debug_ranges": debug_show_ranges
    }
    st.download_button("⬇️ JSON", data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="book_summary.json", mime="application/json")

    md = []
    md.append(f"# {export['meta'].get('title','Book')} — Summary\n")
    md.append("## Executive Summary\n\n" + exec_summary + "\n")
    md.append("## Book Themes\n")
    for t in themes:
        md.append(f"### {t['title']}\n\n" + t["summary"] + "\n\n**Covers:** " + ", ".join(t["chapter_titles"]) + "\n")
    md.append("## Chapters\n")
    for i, c in enumerate(per_chapter, start=1):
        md.append(f"### Chapter {i}: {c['title']}\n\n{c['notes_and_summary']}\n")
    md.append("## Book Glossary\n\n" + glossary_text + "\n")
    md.append("## Claims Index (by Theme)\n")
    for c in claims_by_theme:
        md.append(f"### {c['theme_title']}\n\n{c['claims']}\n")
    st.download_button("⬇️ Markdown", data="".join(md).encode("utf-8"),
                       file_name="book_summary.md", mime="text/markdown")

else:
    st.info("Upload an EPUB or PDF. Chapters are detected via TOC/OPF; the app presents one card per chapter with book-wide themes, glossary, and claims.")
