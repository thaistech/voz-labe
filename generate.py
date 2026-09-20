from http.server import BaseHTTPRequestHandler
import io,json,os,threading,numpy as np,soundfile as sf
MODEL=None
LOCK=threading.Lock()
def get_model():
 global MODEL
 if MODEL is None:
  with LOCK:
   if MODEL is None:
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MODEL=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
 return MODEL
class handler(BaseHTTPRequestHandler):
 def reply(self,code,obj):
  b=json.dumps(obj,ensure_ascii=False).encode()
  self.send_response(code);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self): self.reply(200,{"ok":True,"engine":"Chatterbox Multilingual","language":"pt","processing":"server"})
 def do_POST(self):
  try:
   n=int(self.headers.get("content-length","0"));d=json.loads(self.rfile.read(n) or b"{}");text=(d.get("text") or "").strip()
   if not text: raise ValueError("Digite um texto.")
   if len(text)>300: raise ValueError("Use até 300 caracteres por geração.")
   m=get_model();ref=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","public","thais-reference.wav"))
   w=m.generate(text,language_id="pt",audio_prompt_path=ref,exaggeration=max(.25,min(2,float(d.get("exaggeration",.5)))),temperature=max(.05,min(5,float(d.get("temperature",.8)))),cfg_weight=max(.2,min(1,float(d.get("cfg_weight",.5)))))
   if hasattr(w,"detach"): w=w.detach().cpu().numpy()
   out=io.BytesIO();sf.write(out,np.asarray(w,dtype=np.float32).squeeze(),m.sr,format="WAV",subtype="PCM_16");a=out.getvalue()
   self.send_response(200);self.send_header("Content-Type","audio/wav");self.send_header("Content-Length",str(len(a)));self.end_headers();self.wfile.write(a)
  except Exception as e:self.reply(500,{"error":str(e)})
