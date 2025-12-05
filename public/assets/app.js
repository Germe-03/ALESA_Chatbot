const API_URL = '/chat';

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

let CURRENT_SESSION = null;

function newSessionId(){
  return (self.crypto?.randomUUID?.() || Math.random().toString(36).slice(2));
}

function getSession(){
  // Sitzung nur im Speicher halten: neuer Tab/Reload => neue Session
  if (!CURRENT_SESSION){
    CURRENT_SESSION = newSessionId();
  }
  return CURRENT_SESSION;
}

function resetSessionUI(){
  CURRENT_SESSION = newSessionId();
  const chat = el('#chat');
  chat.innerHTML = '';
  const hello = createMsg({role:'bot', text:
    'Neue Session gestartet.\n' +
    'Stell mir eine Frage zu Produkten, Services oder Dokumenten – ich antworte mit Quellen.'
  });
  chat.appendChild(hello);
  const ta = el('#q');
  if (ta){ ta.value = ''; ta.focus(); }
}

async function ask(question){
  const form = new FormData();
  form.append('session', getSession());
  form.append('message', question);
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
  const newBtn = el('#new-session');

  const hello = createMsg({role:'bot', text:
    'Hallo! Ich bin Reinhard, der KI-Assistent der Firma ALESA.\n' +
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
    ask(q).then(({responses})=>{
      const out = responses && responses.length ? responses : ['(keine Antwort)'];
      for (const r of out){
        chat.appendChild(createMsg({role:'bot', text: String(r)}));
      }
      scrollToBottom();
    }).catch(err=>{
      const bot = createMsg({role:'bot', text: `Fehler: ${err.message}`});
      chat.appendChild(bot);
      scrollToBottom();
    }).finally(()=> setBusy(false));
  }

  btn.addEventListener('click', submit);
  if (newBtn){ newBtn.addEventListener('click', resetSessionUI); }

  // Enter zum Senden, Shift+Enter = Zeilenumbruch; ignoriert IME-Komposition
  const handleEnter = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !(e.isComposing === true)){
      e.preventDefault();
      submit();
    }
  };
  ta.addEventListener('keydown', handleEnter);
  // Fallback: global listener, falls keydown auf dem textarea nicht feuert
  window.addEventListener('keydown', (e) => {
    if (document.activeElement === ta) {
      handleEnter(e);
    }
  });
}

window.addEventListener('DOMContentLoaded', init);
