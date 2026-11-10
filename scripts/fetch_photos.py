"""抓取知识库中真实地点的实景照片到 static/photos/。

用法：
    python scripts/fetch_photos.py                 # 只补齐缺失照片
    python scripts/fetch_photos.py --force         # 全部重抓
    python scripts/fetch_photos.py --only p01 r02  # 指定实体
    python scripts/fetch_photos.py --report        # 只查看覆盖情况

策略：
1. 以地点名称（餐饮带菜系、酒店带"酒店"）检索公开图片，标题必须命中地点名，
   并优先选择标题里出现"深圳/所在区/攻略·门票·探店"等词的候选，避免同名异地；
2. 下载后裁切为 4:3、宽 800px 的 JPEG，并裁掉图库截图常见的纯色留白边；
3. 同一张图不会分配给两个地点（md5 去重）；
4. 找不到可靠实景图的地点**不生成图片**，前端退回排版占位——
   不使用 AI 生成图，也不用无关配图充数。

结果写入 static/photos/manifest.json（含标题与来源页），图片仅用于本地演示。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.shenzhen import HOTELS, POIS, RESTAURANTS  # noqa: E402

PHOTO_DIR = os.path.join(ROOT, "static", "photos")
THUMB_DIR = os.path.join(PHOTO_DIR, "thumbs")
MANIFEST = os.path.join(PHOTO_DIR, "manifest.json")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TARGET = (800, 600)
MIN_BYTES = 5000

# 个别名称在图片标题里写法不同，这里给出检索式与标题匹配词
OVERRIDES: dict[str, tuple[str, str]] = {
    "p01": ("深圳湾公园 海滨栈道 日落", "深圳湾公园"),
    "p17": ("地王大厦 深圳 观光", "地王大厦"),
    "p35": ("欢乐海岸 深圳 曲水湾 水秀", "欢乐海岸"),
    "p57": ("cocopark 深圳 购物中心", "cocopark"),
    "r18": ("大梅沙京基海湾酒店 自助餐", "京基海湾"),
    "h05": ("前海 JW万豪酒店 深圳", "jw万豪"),
}

OTHER_CITIES = (
    "北京", "上海", "广州", "成都", "重庆", "武汉", "杭州", "西安", "长沙", "南京",
    "天津", "苏州", "青岛", "厦门", "昆明", "大连", "宁波", "郑州", "济南", "合肥",
    "福州", "南宁", "贵阳", "太原", "石家庄", "哈尔滨", "沈阳", "长春", "南昌",
    "兰州", "乌鲁木齐", "呼和浩特", "银川", "西宁", "拉萨", "海口", "三亚",
    "香港", "澳门", "台北", "佛山", "东莞", "珠海", "惠州", "汕头",
)
BAD_WORDS = (
    "营收", "业绩", "股价", "财报", "融资", "上市", "招聘", "加盟", "峰会", "论坛",
    "发布会", "签约", "战略合作", "裁员", "市值", "招股", "地图", "导航图",
    "位置图", "交通图", "示意图", "平面图", "户型图", "海报", "展板", "简笔画",
)
GOOD_WORDS = ("门票", "攻略", "游玩", "实景", "景区", "景点", "探店", "美食",
              "餐厅", "酒店", "打卡", "体验", "游记")


def _get(url: str, referer: str = "https://image.baidu.com/", timeout: int = 25):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": referer,
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def core_name(name: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", name).strip()


def search_images(query: str, pages: int = 1) -> list[dict]:
    out: list[dict] = []
    for page in range(pages):
        url = (
            "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592"
            "&is=&fp=result&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&nc=1"
            "&queryWord=" + urllib.parse.quote(query)
            + "&word=" + urllib.parse.quote(query)
            + f"&pn={page * 30}&rn=30&gsm=1e"
        )
        try:
            _, body, _ = _get(url)
            data = json.loads(body.decode("utf-8", "ignore").replace("\\'", "'"))
        except Exception:
            break
        items = [it for it in (data.get("data") or []) if it]
        for it in items:
            replace = (it.get("replaceUrl") or [{}])[0] or {}
            out.append({
                "title": re.sub(r"</?strong>", "", it.get("fromPageTitleEnc") or ""),
                "mid": it.get("middleURL") or it.get("thumbURL") or "",
                "obj": replace.get("ObjURL", ""),
                "page": it.get("fromURL") or "",
            })
        if len(items) < 30:
            break
        time.sleep(0.2)
    return out


def score_title(kind: str, name: str, title: str, area: str, cuisine: str = "",
                match: str = "") -> int:
    t = title.replace(" ", "")
    if not t or any(w in t for w in BAD_WORDS):
        return 0
    core = (match or core_name(name)).replace(" ", "").lower()
    t = t.lower()
    has_city = "深圳" in t or (area and area in t)
    if not has_city and any(c in t for c in OTHER_CITIES):
        return 0
    score = 0
    if core and core in t:
        score += 3
    elif kind == "restaurant" and cuisine:
        first = re.split(r"[/、·]", cuisine)[0].replace(" ", "")
        if len(first) >= 2 and first in t:
            score += 1
        else:
            return 0
    else:
        return 0
    if has_city:
        score += 2
    if any(w in t for w in GOOD_WORDS):
        score += 1
    return score


def queries_for(kind: str, name: str, area: str, cuisine: str = "") -> list[str]:
    core = core_name(name)
    if kind == "restaurant":
        return [f"{core} 深圳", f"{core} {cuisine}".strip(), core]
    if kind == "hotel":
        return [core, f"{core} 大堂"]
    return [core, f"{core} 深圳", f"{core} 实景"]


def _trim_border(img: Image.Image) -> Image.Image:
    """裁掉纯白/纯黑的图库留白边。"""
    gray = img.convert("L")
    w, h = gray.size
    px = gray.load()

    def row_uniform(y: int) -> bool:
        vals = [px[x, y] for x in range(0, w, max(1, w // 60))]
        return (sum(1 for v in vals if v > 244) > len(vals) * 0.97
                or sum(1 for v in vals if v < 12) > len(vals) * 0.97)

    def col_uniform(x: int) -> bool:
        vals = [px[x, y] for y in range(0, h, max(1, h // 60))]
        return (sum(1 for v in vals if v > 244) > len(vals) * 0.97
                or sum(1 for v in vals if v < 12) > len(vals) * 0.97)

    top, bottom, left, right = 0, h, 0, w
    while top < h // 4 and row_uniform(top):
        top += 1
    while bottom > h * 3 // 4 and row_uniform(bottom - 1):
        bottom -= 1
    while left < w // 4 and col_uniform(left):
        left += 1
    while right > w * 3 // 4 and col_uniform(right - 1):
        right -= 1
    if (right - left) < w * 0.6 or (bottom - top) < h * 0.6:
        return img
    return img.crop((left, top, right, bottom))


def normalize(raw: bytes) -> bytes | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None
    img = _trim_border(img.convert("RGB"))
    if img.width < 440 or img.height < 300:
        return None
    tw, th = TARGET
    scale = max(tw / img.width, th / img.height)
    resized = img.resize(
        (max(tw, int(img.width * scale + 0.5)), max(th, int(img.height * scale + 0.5))),
        Image.LANCZOS,
    )
    left = (resized.width - tw) // 2
    top = int((resized.height - th) * 0.35)
    out = io.BytesIO()
    resized.crop((left, top, left + tw, top + th)).save(out, "JPEG", quality=82, optimize=True)
    data = out.getvalue()
    return data if len(data) >= MIN_BYTES else None


def write_thumb(data: bytes, eid: str) -> None:
    """额外生成 360x270 缩略图，供界面卡片使用，减少每次渲染的传输体积。"""
    try:
        os.makedirs(THUMB_DIR, exist_ok=True)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((360, 270), Image.LANCZOS)
        img.save(os.path.join(THUMB_DIR, f"{eid}.jpg"), "JPEG", quality=80, optimize=True)
    except Exception:
        pass


def build_thumbs() -> int:
    os.makedirs(THUMB_DIR, exist_ok=True)
    count = 0
    for entry in os.listdir(PHOTO_DIR):
        if not entry.endswith(".jpg"):
            continue
        eid = entry[:-4]
        thumb = os.path.join(THUMB_DIR, f"{eid}.jpg")
        if os.path.exists(thumb):
            continue
        with open(os.path.join(PHOTO_DIR, entry), "rb") as f:
            write_thumb(f.read(), eid)
        count += 1
    return count


def fetch_one(item: dict, force: bool, used: dict[str, str]) -> tuple[str, str, dict | None]:
    eid, name, kind = item["id"], item["name"], item["kind"]
    area, cuisine = item.get("area", ""), item.get("cuisine", "")
    path = os.path.join(PHOTO_DIR, f"{eid}.jpg")
    if os.path.exists(path) and not force:
        return eid, "skip", None

    override = OVERRIDES.get(eid)
    queries = [override[0]] if override else queries_for(kind, name, area, cuisine)
    match = override[1] if override else ""

    ranked: list[tuple[int, dict, str]] = []
    seen: set[str] = set()
    for query in queries:
        for cand in search_images(query):
            if cand["mid"] in seen or not cand["mid"]:
                continue
            seen.add(cand["mid"])
            sc = score_title(kind, name, cand["title"], area, cuisine, match)
            if sc >= 3:
                ranked.append((sc, cand, query))
        if ranked and max(r[0] for r in ranked) >= 5:
            break

    ranked.sort(key=lambda r: -r[0])
    for _, cand, query in ranked[:6]:
        for url in (cand["mid"], cand["obj"]):
            if not url:
                continue
            try:
                status, raw, ctype = _get(url, referer=cand["page"] or "https://image.baidu.com/")
            except Exception:
                continue
            if status != 200 or "image" not in ctype or len(raw) < MIN_BYTES:
                continue
            data = normalize(raw)
            if not data:
                continue
            digest = hashlib.md5(data).hexdigest()
            if digest in used:
                continue
            with open(path, "wb") as f:
                f.write(data)
            write_thumb(data, eid)
            used[digest] = eid
            return eid, "ok", {
                "name": name,
                "kind": kind,
                "query": query,
                "title": cand["title"],
                "source": cand["mid"],
                "page": cand["page"],
                "bytes": len(data),
                "md5": digest,
            }
    return eid, "miss", None


def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--thumbs", action="store_true", help="只为已有大图补生成缩略图")
    args = ap.parse_args()

    os.makedirs(PHOTO_DIR, exist_ok=True)
    if args.thumbs:
        print(f"生成缩略图 {build_thumbs()} 张")
        return
    manifest = load_manifest()

    items = (
        [{"id": p.id, "name": p.name, "kind": "poi", "area": p.area} for p in POIS]
        + [{"id": r.id, "name": r.name, "kind": "restaurant", "area": r.area, "cuisine": r.cuisine}
           for r in RESTAURANTS]
        + [{"id": h.id, "name": h.name, "kind": "hotel", "area": h.area} for h in HOTELS]
    )
    if args.only:
        wanted = set(args.only)
        items = [i for i in items if i["id"] in wanted]

    have = sum(1 for i in items if os.path.exists(os.path.join(PHOTO_DIR, f"{i['id']}.jpg")))
    if args.report:
        print(f"图片覆盖：{have}/{len(items)}")
        for i in items:
            mark = "有" if os.path.exists(os.path.join(PHOTO_DIR, f"{i['id']}.jpg")) else "无"
            print(f"  {mark}  {i['id']:<5} {i['name']}")
        return
    if have == len(items) and not args.force:
        print(f"图片已齐全：{have}/{len(items)}")
        return

    used: dict[str, str] = {}
    for entry in os.listdir(PHOTO_DIR):
        if entry.endswith(".jpg"):
            with open(os.path.join(PHOTO_DIR, entry), "rb") as f:
                used.setdefault(hashlib.md5(f.read()).hexdigest(), entry[:-4])

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "miss": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, it, args.force, used): it for it in items}
        for fut in futures:
            item = futures[fut]
            try:
                eid, state, meta = fut.result()
            except Exception as exc:
                print(f"[err]  {item['id']} {item['name']}: {type(exc).__name__}")
                stats["miss"] += 1
                continue
            stats[state] += 1
            if state == "ok":
                manifest[eid] = meta
                print(f"[ok]   {eid:<5} {meta['name']}  ← {meta['title'][:36]}")
            elif state == "miss":
                print(f"[miss] {eid:<5} {item['name']}")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    thumbs = build_thumbs()
    if thumbs:
        print(f"生成缩略图 {thumbs} 张")
    total = len([p for p in os.listdir(PHOTO_DIR) if p.endswith(".jpg")])
    print(f"\n完成：新增 {stats['ok']} / 跳过 {stats['skip']} / 未命中 {stats['miss']}，"
          f"本地共 {total} 张，耗时 {time.time() - t0:.1f}s")
    print("未命中地点在界面上使用排版占位，不使用 AI 图或无关配图。")


if __name__ == "__main__":
    main()
