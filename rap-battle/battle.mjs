import {createPerformer} from "./model.mjs";

const find = id => document.getElementById(id);
const names = ["Laptop", "Cloud"];
const rounds = 3;
const totalTurns = names.length * rounds;
const verses = find("verses");
const choices = find("choices");
const next = find("next");
const reset = find("reset");
const status = find("status");
const common = new Set("the and your you that when with from this have their then were them like there what they will just into back while still whole already".split(" "));
const cueNouns = new Set((
  "battery server racks broadband cable mic tunnel bike queue curb beat concert seat echo passport keys " +
  "rhythm signal fan band bicycle spinner dance verse chart towers pocket punchline skyline invoice solo " +
  "screen gate backpack speakers chorus room center bell spark stage lights copy crown case music memory " +
  "shortcut track meter network keyboard page show packets truck router ticket rain train zone record " +
  "cover chrome charger cabin silence grid fire bass stack studio notebook snow chart drum coast lid " +
  "cloth circuit credit workload laptop candle park wings crew sprint relay moon table file desk warehouse " +
  "wire tempo crash badge team stream flight hooks trees park water hall crowd door tunnel data time sound"
).split(" "));
let models;
let turn = 0;
let previous = "";
let cue = "";
let busy = false;

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function cueWords(text) {
  const words = [...new Set(text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g))];
  const candidates = words.filter(word => word.length > 3 && !common.has(word));
  const nouns = candidates.filter(word => cueNouns.has(word));
  return (nouns.length ? nouns : candidates.length ? candidates : words).slice(0, 3);
}

async function loadModels() {
  return Promise.all(names.map(async name => {
    const response = await fetch(new URL(name.toLowerCase() + ".json", import.meta.url));
    if (!response.ok) throw new Error("Model download failed.");
    return createPerformer(await response.json());
  }));
}

function appendVerse(performer, text) {
  const item = element("li", "");
  item.className = "verse " + performer.toLowerCase();
  item.append(element("h3", performer + " · Round " + (Math.floor(turn / names.length) + 1)));
  const verse = element("p", "");
  let found = false;
  for (const piece of text.split(/([A-Za-z]+(?:'[A-Za-z]+)?)/)) {
    if (cue && piece.toLowerCase() === cue) {
      found = true;
      verse.append(element("mark", piece));
    } else {
      verse.append(piece);
    }
  }
  item.append(verse);
  if (cue) {
    const detail = element("p", found ? "Callback · " + cue : "Missed callback · " + cue);
    detail.className = "callback";
    item.append(detail);
  }
  verses.append(item);
}

function showChoices() {
  choices.replaceChildren();
  const performer = names[turn % names.length];
  const words = cueWords(previous);
  status.textContent = words.length ? "Pick a word for " + performer + " to answer." : performer + " takes the mic.";
  for (const word of words) {
    const button = element("button", word);
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.onclick = () => {
      cue = word;
      for (const choice of choices.children) {
        choice.setAttribute("aria-pressed", String(choice === button));
      }
      next.disabled = false;
    };
    choices.append(button);
  }
  next.textContent = "Generate " + performer + "’s verse";
  next.disabled = words.length > 0;
  (choices.firstElementChild ?? next).focus();
}

function showVote() {
  find("round").textContent = "Battle complete";
  status.textContent = "Who won?";
  next.hidden = true;
  choices.setAttribute("aria-label", "Vote for the winner");
  for (const name of names) {
    const button = element("button", name);
    button.type = "button";
    button.onclick = () => {
      choices.replaceChildren();
      status.textContent = "You voted for " + name + ".";
      reset.focus();
    };
    choices.append(button);
  }
  choices.firstElementChild.focus();
}

next.onclick = async () => {
  if (busy) return;
  busy = next.disabled = reset.disabled = true;
  const performer = names[turn % names.length];
  status.textContent = models ? performer + " is writing…" : "Loading Laptop and Cloud…";
  try {
    models ??= await loadModels();
    const stats = find("model-stats");
    stats.textContent = models.map((model, index) => names[index] + " · " + model.training.parameters.toLocaleString() + " trained parameters").join(" / ");
    stats.hidden = false;
    // Give the browser a frame to announce the busy state before inference.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const text = models[turn % names.length].generate({previous, cue, seed});
    if (!text) throw new Error("The model returned an empty verse.");
    appendVerse(performer, text);
    previous = text;
    cue = "";
    turn++;
    reset.hidden = false;
    choices.replaceChildren();
    if (turn === totalTurns) {
      showVote();
    } else {
      find("round").textContent = "Round " + (Math.floor(turn / names.length) + 1) + " of " + rounds;
      showChoices();
    }
  } catch {
    status.textContent = models ? "The verse could not be generated. Try this turn again." : "Could not load both models. Check your connection and try again.";
    next.disabled = false;
    next.focus();
  } finally {
    busy = reset.disabled = false;
  }
};

reset.onclick = () => {
  turn = 0;
  previous = cue = "";
  verses.replaceChildren();
  choices.replaceChildren();
  choices.setAttribute("aria-label", "Choose a callback");
  find("round").textContent = "Round 1 of 3";
  status.textContent = "Both models are ready.";
  next.textContent = "Start battle";
  next.hidden = next.disabled = false;
  reset.hidden = true;
  next.focus();
};
