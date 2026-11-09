#!/usr/bin/env python3
"""Rebuild, commit website updates, and push using the device's Git credentials."""
import argparse, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def run(*args,**kwargs): return subprocess.run(args,cwd=ROOT,check=True,**kwargs)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--message',default='Update project results and videos'); args=p.parse_args()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='main': raise SystemExit('Switch to main before publishing.')
    run(sys.executable,'scripts/build_site.py')
    run(sys.executable,'-m','unittest','discover','-s','tests')
    run('git','add','README.md','THIRD_PARTY_NOTICES.md','.gitignore','.github','scripts','templates','assets','data','videos','tests','index.html')
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    if changed==1: run('git','commit','-m',args.message)
    elif changed!=0: raise SystemExit('Unable to inspect staged changes.')
    run('git','push','origin','main')
if __name__=='__main__': main()
