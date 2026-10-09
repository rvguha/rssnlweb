"""Which shows publish transcripts, and how are they shaped? For each target show, fetch three episodes (recent, middling, old), try the
npr.org transcript page (or the episode page) and report the transcript's length, speaker-labelled lines and timestamps.

    python scripts/probe_transcripts.py        # from the repo root; writes probe_tx.json

Findings of 2026-10-09 are in the board note "NPR and NYT transcript candidates": Weekend Edition, Up First, Consider This, Throughline,
Invisibilia and TED Radio Hour have speaker-labelled transcripts on npr.org with no timestamps; Fresh Air has none; nytimes.com refuses scripts.
"""
import re,html,json,urllib.request,concurrent.futures as cf
raw=open('sources.yaml').read()
feeds={n:u for n,u in re.findall(r"- name: (\S+)\n  url: (\S+)\n",raw)}
targets={"Fresh Air":feeds["npr_fresh_air"],"TED Radio Hour":feeds["npr_ted_radio_hour"],"Throughline":feeds["npr_throughline"],"Radiolab":feeds["npr_radiolab"],
 "99% Invisible":feeds["npr_99_invisible"],"Invisibilia":feeds["npr_invisibilia"],"Up First":feeds["npr_up_first_from_npr"],"On the Media":feeds["npr_on_the_media"],
 "Morning Edition":"https://feeds.npr.org/510318/podcast.xml","All Things Considered":"https://feeds.npr.org/510313/podcast.xml","Weekend Edition Saturday":"https://feeds.npr.org/510317/podcast.xml",
 "Serial (NYT)":feeds["nyt_serial"],"Nice White Parents":feeds["nyt_nice_white_parents"],"The Daily":feeds["nyt_the_daily"]}
UA={"User-Agent":"Mozilla/5.0 (compatible; research-probe)"}
def get(u,t=30): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t).read().decode("utf-8","replace")
def body(s):
    s=re.sub(r"<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<header.*?</header>|<footer.*?</footer>","",s,flags=re.S)
    return html.unescape(re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",s)))
def one(item):
    name,u=item; out={"show":name}
    try: x=get(u)
    except Exception as e: out["error"]=str(e)[:50]; return out
    its=re.findall(r"<item>(.*?)</item>",x,re.S); out["episodes_in_feed"]=len(its)
    tests=[]
    for idx in (5,30,min(150,len(its)-1)):
        if idx>=len(its) or idx<0: continue
        it=its[idx]; link=(re.findall(r"<link>(.*?)</link>",it,re.S) or [""])[0].strip()
        title=html.unescape(re.sub(r"<!\[CDATA\[|\]\]>","",(re.findall(r"<title>(.*?)</title>",it,re.S) or [""])[0]))[:60]
        date=(re.findall(r"<pubDate>(.*?)</pubDate>",it) or [""])[0][:16]
        cands=[link]
        m=re.search(r"npr\.org/(?:\d{4}/\d\d/\d\d/)?(nx-s1-\d+|\d{8,})",link) 
        if m: cands.insert(0,f"https://www.npr.org/transcripts/{m.group(1)}")
        best=None
        for cu in cands:
            if not cu: continue
            try:
                t=body(get(cu))
            except Exception as e:
                best=best or {"url":cu,"error":str(e)[:40]}; continue
            sp=len(re.findall(r"(?:^| )[A-Z][A-Z .'’-]{2,28}:",t)); ts=len(re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b",t))
            cand={"url":cu,"chars":len(t),"speaker_labels":sp,"timestamps":ts,"mentions":len(re.findall(r"transcript",t,re.I))}
            if not best or cand["chars"]>best.get("chars",0): best=cand
        tests.append({"title":title,"date":date,**(best or {})})
    out["tests"]=tests; return out
with cf.ThreadPoolExecutor(6) as ex: res=list(ex.map(one,targets.items()))
json.dump(res,open("probe_tx.json","w"),indent=1)
for r in res:
    print(r["show"],r.get("episodes_in_feed"),r.get("error",""))
    for t in r.get("tests",[]): print("   ",t["date"],t["title"][:40],"|",t.get("chars"),"chars spk",t.get("speaker_labels"),"ts",t.get("timestamps"),t.get("error",""))
