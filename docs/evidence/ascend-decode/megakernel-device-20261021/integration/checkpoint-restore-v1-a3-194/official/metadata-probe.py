import os,pathlib,subprocess,json
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=pathlib.Path('/home/shanshan/tilert-megakernel-20261021/checkpoint-owned/evidence'); p.mkdir(parents=True,exist_ok=True)
u='https://modelscope.cn/api/v1/models/Eco-Tech/GLM-5.2-w4a8c8/repo/files?Revision=67e2b7157ac7d8ce1d15d7251853ce146819ceb9&Recursive=true'
c=['curl','--fail','--location','--silent','--show-error','--connect-timeout','10','--max-time','45','--dump-header',str(p/'official-files.headers'),'--output',str(p/'official-files.json'),u]
r=subprocess.run(c,capture_output=True,text=True)
(p/'official-files-command.json').write_text(json.dumps({'argv':c,'returncode':r.returncode,'stderr':r.stderr},indent=2)+'\n')
print('returncode',r.returncode,'bytes',(p/'official-files.json').stat().st_size if (p/'official-files.json').exists() else None)
