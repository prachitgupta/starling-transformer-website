const katex=require('katex');const fs=require('fs');
let s=fs.readFileSync(process.argv[2],'utf8');
const dec=t=>t.replace(/&amp;/g,'&');
s=s.replace(/\\\[([\s\S]*?)\\\]/g,(m,t)=>katex.renderToString(dec(t),{displayMode:true,throwOnError:true}));
s=s.replace(/\\\(([\s\S]*?)\\\)/g,(m,t)=>katex.renderToString(dec(t),{throwOnError:true}));
fs.writeFileSync(process.argv[3],s);
