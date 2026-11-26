"""Local interactive demo. Run: python app.py, then open http://127.0.0.1:8765.

Uses Python's HTTP server to avoid an additional web-framework dependency.
All user data stays within the local process; each browser owns its own state.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
from urllib.parse import parse_qs, urlparse

import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from recommendation import CarRecommender, Rule

ROOT=PROJECT_ROOT
config_path=ROOT/'results'/'metadata.json'
config=json.loads(config_path.read_text(encoding='utf-8'))['selected'] if config_path.exists() else {}
ENGINE=CarRecommender(ROOT/'data'/'CAR DETAILS FROM CAR DEKHO.csv',weights=config.get('weights'),alpha=config.get('alpha',0))


def sanitize(value):
    if isinstance(value,dict): return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [sanitize(v) for v in value]
    if isinstance(value,Rule): return value.to_dict()
    if hasattr(value,'item'): return value.item()
    return value


def dispatch(path,payload):
    current=ENGINE.item(int(payload.get('current_id',2)))
    rules=[Rule.from_dict(x) for x in payload.get('rules',[])]
    exclude=[int(payload.get('start_id',current['case_id']))]
    if path=='/api/recommend':
        rows=ENGINE.rank(current,rules,k=int(payload.get('k',10)),exclude=exclude)
        for row in rows.to_dict('records'):
            pass
        return dict(items=rows.to_dict('records'),candidate_count=int(ENGINE.mask(rules,exclude).sum()),
                    current=current,rules=[r.describe() for r in rules])
    if path=='/api/critique':
        feedback=[ENGINE.critique_rule(current,x['field'],x.get('value')) for x in payload.get('feedback',[])]
        return dict(rules=ENGINE.merge_rules(rules,feedback))
    if path=='/api/dynamic':
        return dict(patterns=ENGINE.dynamic_critiques(current,rules,exclude,limit=8))
    if path=='/api/repair':
        return dict(proposals=ENGINE.repair_proposals(current,rules,exclude))
    raise ValueError('Unknown endpoint')


class Handler(BaseHTTPRequestHandler):
    def respond(self,data,status=200,mime='application/json; charset=utf-8'):
        blob=data.encode('utf-8') if isinstance(data,str) else json.dumps(sanitize(data),ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(blob)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path=='/':
            return self.respond((ROOT/'website'/'index.html').read_text(encoding='utf-8'),mime='text/html; charset=utf-8')
        if parsed.path=='/api/catalog':
            query=parse_qs(parsed.query).get('q',[''])[0]
            rows=ENGINE.data[ENGINE.data.name.str.contains(query,case=False,regex=False)].head(50)
            return self.respond(dict(items=rows.to_dict('records'),total=len(ENGINE.data),
                    categories={c:sorted(ENGINE.data[c].unique().tolist()) for c in ('fuel','transmission','brand')}))
        return self.respond(dict(error='Not found'),404)

    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length>100000: return self.respond(dict(error='Request too large'),413)
            payload=json.loads(self.rfile.read(length))
            return self.respond(dispatch(self.path,payload))
        except (ValueError,KeyError,TypeError) as error:
            return self.respond(dict(error=str(error)),400)

    def log_message(self,format,*args):
        print(format % args,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Demo khuyến nghị xe cũ CBR')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
