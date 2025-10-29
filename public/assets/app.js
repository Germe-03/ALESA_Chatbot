const API_URL = '/ask';

const el = (sel, root=document) => root.querySelector(sel);
const els = (sel, root=document) => Array.from(root.querySelectorAll(sel));

function createMsg({role, text, sources=[]}){
  const wrap = document.createElement('div');
  wrap.className = 'bubble';
  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role==='bot' ? 'bot':'user');
  avatar.textContent = role==='bot' ? 'A' : 'Du';
  const msg = document.createElement('div');
  msg.className = 'msg';
  msg.innerHTML = text.split('\n').map(p=>`<p>${escapeHtml(p)}</p>`).join('');
  if (sources && sources.length){
    const src = document.createElement('div');
    src.className = 'sources';
    src.innerHTML = `<h4>Quellen</h4>` + sources.map(s=>`<div class="tag">${escapeHtml(s)}</div>`).join('');
    msg.appendChild(src);
  }
  wrap.append(avatar, msg);
  return wrap;
}

function escapeHtml(s){
  return (s||'')
   .replaceAll('&','&amp;')
   .replaceAll('<','&lt;')
   .replaceAll('>','&gt;');
}

async function ask(question){
  const form = new FormData();
  form.append('question', question);
  const res = await fetch(API_URL, {method:'POST', body: form});
  if (!res.ok){
    const t = await res.text().catch(()=> '');
    throw new Error(`Fehler ${res.status}: ${t||res.statusText}`);
  }
  return await res.json();
}

function scrollToBottom(){
  window.scrollTo({top: document.body.scrollHeight, behavior:'smooth'});
}

function setBusy(b){
  el('#send').disabled = b;
  el('#q').disabled = b;
}

function init(){
  const chat = el('#chat');
  const ta = el('#q');
  const btn = el('#send');

  const hello = createMsg({role:'bot', text:
    'Hallo! Ich bin ALESA, dein virtueller KI‑Assistent.\n' +
    'Stell mir eine Frage zu Produkten, Services oder Dokumenten – ich antworte mit Quellen.'
  });
  chat.appendChild(hello);

  function submit(){
    const q = (ta.value||'').trim();
    if (!q) return;
    el('#empty')?.remove();
    const user = createMsg({role:'user', text:q});
    chat.appendChild(user);
    ta.value='';
    setBusy(true);
    scrollToBottom();
    ask(q).then(({answer, sources})=>{
      const bot = createMsg({role:'bot', text: answer || 'Keine Antwort erhalten.', sources: sources||[]});
      chat.appendChild(bot);
      scrollToBottom();
    }).catch(err=>{
      const bot = createMsg({role:'bot', text: `Fehler: ${err.message}`});
      chat.appendChild(bot);
      scrollToBottom();
    }).finally(()=> setBusy(false));
  }

  btn.addEventListener('click', submit);
  ta.addEventListener('keydown', (e)=>{
    if (e.key==='Enter' && !e.shiftKey){
      e.preventDefault(); submit();
    }
  });
}

window.addEventListener('DOMContentLoaded', init);

