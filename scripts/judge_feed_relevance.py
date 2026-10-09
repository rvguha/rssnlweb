"""Have a model judge how many of a feed query's results are relevant to it.

    OPENROUTER_API_KEY=... python scripts/judge_feed_relevance.py COLLECTION QUERIES.json     # a JSON list of query strings

Prints relevant/returned for each query and writes QUERIES.json.out. The same judgment, with a source-variety measure, is part of
NeuralKG2's tools/status_ledger.py.
"""
import re,json,os,sys,urllib.parse,urllib.request,concurrent.futures as cf,html
# usage: judge.py COLLECTION QUERIES.json   (a JSON list of query strings)  -> prints relevant/total per query
COLLECTION=sys.argv[1]; QS=json.load(open(sys.argv[2]))
def feed(q):
    u="https://rss.neuralweb.dev/feed.xml?"+urllib.parse.urlencode({"q":q,"collection":COLLECTION,"limit":"40","since":"2000-01-01"})
    x=urllib.request.urlopen(u,timeout=90).read().decode()
    out=[]
    for it in re.findall(r"<item>(.*?)</item>",x,re.S):
        t=html.unescape(re.sub(r"<!\[CDATA\[|\]\]>","",re.findall(r"<title>(.*?)</title>",it,re.S)[0])).strip()
        d=re.findall(r"<description>(.*?)</description>",it,re.S)
        d=html.unescape(re.sub(r"<[^>]+>|<!\[CDATA\[|\]\]>"," ",html.unescape(d[0]) if d else ""))
        out.append({"title":t,"desc":re.sub(r"\s+"," ",d)[:240]})
    return out
def ask(prompt):
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",json.dumps({"model":"anthropic/claude-opus-5-5","messages":[{"role":"user","content":prompt}]}).encode(),{"Authorization":"Bearer "+os.environ["OPENROUTER_API_KEY"],"Content-Type":"application/json"})
    for _ in range(3):
        text=json.loads(urllib.request.urlopen(req,timeout=180).read())["choices"][0]["message"]["content"] or ""
        m=re.search(r"\{.*\}",text,re.S)
        if m:
            try: return json.loads(m.group(0))
            except ValueError: pass
    return {"relevant":[],"note":"judge gave no usable reply"}
def judge(q):
    items=feed(q)
    if not items: return q,0,0,[]
    listing="\n".join(f"{i}. {x['title']} — {x['desc']}" for i,x in enumerate(items[:40]))
    r=ask('A reader subscribes to a podcast feed defined by this standing interest:\n"%s"\nFor each episode below say whether it is relevant to that interest (it would belong in the feed). Return {"relevant":[indexes],"note":"one sentence on overall quality"}.\n\n%s'%(q,listing))
    rel=[i for i in r.get("relevant",[]) if isinstance(i,int)]
    return q,len(items),len(rel),r.get("note","")
with cf.ThreadPoolExecutor(4) as ex:
    res=list(ex.map(judge,QS))
json.dump(res,open(sys.argv[2]+".out","w"),indent=1)
for q,n,rel,note in res: print(f"{rel:>2}/{n:<2} {q[:100]} | {str(note)[:90]}")
