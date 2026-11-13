import {createPerformer} from "./model.mjs";

const find = id => document.getElementById(id);
const roster = {
  laptop: "Laptop",
  cloud: "Cloud",
  robot: "Robot",
  toaster: "Toaster",
  pigeon: "Pigeon",
  raccoon: "Raccoon",
  librarian: "Librarian",
  "haunted-printer": "Haunted Printer",
  "disco-ball": "Disco Ball",
  "grandfather-clock": "Grandfather Clock",
  cactus: "Cactus",
  octopus: "Octopus",
  "weather-forecaster": "Weather Forecaster",
  volcano: "Volcano",
};
const matchup = find("matchup");
const selectors = [find("first-opponent"), find("second-opponent")];
const modelLoads = new Map();
const rounds = 3;
const totalTurns = selectors.length * rounds;
const verses = find("verses");
const choices = find("choices");
const controls = choices.parentElement;
const next = find("next");
const reset = find("reset");
const temperature = find("temperature");
const temperatureValue = find("temperature-value");
const status = find("status");
const common = new Set("the and your you that when with from this have their then were them like there what they will just into back while still whole already".split(" "));
const cueNouns = new Set((
  "battery server racks broadband cable mic tunnel bike queue curb beat concert seat echo passport keys " +
  "rhythm signal fan band bicycle spinner dance verse chart towers pocket punchline skyline invoice solo " +
  "screen gate backpack speakers chorus room center bell spark stage lights copy crown case music memory " +
  "shortcut track meter network keyboard page show packets truck router ticket rain train zone record " +
  "cover chrome charger cabin silence grid fire bass stack studio notebook snow chart drum coast lid " +
  "cloth circuit credit workload laptop candle park wings crew sprint relay moon table file desk warehouse " +
  "wire tempo crash badge team stream flight hooks trees park water hall crowd door tunnel data time sound " +
  "gears lever crumbs slot bread hinge slice sensors coils elbow cord motor dial timer wheels spring steel " +
  "fountain beak rooftop flock lamppost gutter picnic bucket alley horn latch shelf footnote index spine " +
  "toner paper tray rollers ghost receipt margin glitter mirror mirrors sparkle spotlight chain pendulum " +
  "chime chimes floorboards thorn needles sand desert bloom shadow reef shell tide ink kelp radar forecast " +
  "drizzle thunder broadcast crater lava ash smoke rock ridge slope"
).split(" "));
let replyModel;
let challengeModel;
const performed = selectors.map(() => []);
const selectedNames = () => selectors.map(select => roster[select.value]);
let turn = 0;
let previousChallenge = "";
let cue = "";
let busy = false;

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function updateTemperature() {
  temperatureValue.value = temperature.valueAsNumber.toFixed(2);
}

temperature.addEventListener("input", updateTemperature);
updateTemperature();

function cueWords(text) {
  const words = [...new Set(text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g))];
  const candidates = words.filter(word => word.length > 3 && !common.has(word));
  const nouns = candidates.filter(word => cueNouns.has(word));
  return [...new Set([...nouns, ...candidates, ...words])].slice(0, 3);
}

function updateMatchup() {
  selectors.forEach((select, index) => {
    const other = selectors[1 - index];
    for (const option of select.options) option.disabled = option.value === other.value;
  });
}

selectors.forEach((select, index) => {
  for (const [id, name] of Object.entries(roster)) {
    const option = element("option", name);
    option.value = id;
    select.append(option);
  }
  select.value = index === 0 ? "laptop" : "cloud";
  select.addEventListener("change", updateMatchup);
});
updateMatchup();

function loadModel(id) {
  if (!modelLoads.has(id)) {
    const pending = fetch(new URL(id + ".json", import.meta.url)).then(async response => {
      if (!response.ok) throw new Error("Model download failed.");
      return createPerformer(await response.json());
    }).catch(error => {
      modelLoads.delete(id);
      throw error;
    });
    modelLoads.set(id, pending);
  }
  return modelLoads.get(id);
}

function appendVerse(performer, reply, challenge) {
  const item = element("li", "");
  item.className = turn % selectors.length === 0 ? "verse first" : "verse second";
  item.append(element("h3", performer + " · Round " + (Math.floor(turn / selectors.length) + 1)));
  const verse = element("p", "");
  if (turn === 0) item.append(element("h4", "Opening"));
  for (const piece of reply.split(/([A-Za-z]+(?:'[A-Za-z]+)?)/)) {
    if (cue && piece.toLowerCase() === cue) {
      verse.append(element("mark", piece));
    } else {
      verse.append(piece);
    }
  }
  item.append(verse, element("p", challenge));
  verses.append(item);
}

function showChoices() {
  choices.replaceChildren();
  const performer = selectedNames()[turn % selectors.length];
  const words = cueWords(previousChallenge);
  status.textContent = words.length ? "Pick a word from the challenge for " + performer + " to answer." : performer + " takes the mic.";
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

next.onclick = async () => {
  if (busy) return;
  busy = next.disabled = reset.disabled = temperature.disabled = true;
  const samplingTemperature = temperature.valueAsNumber;
  for (const choice of choices.children) choice.disabled = true;
  matchup.disabled = true;
  const names = selectedNames();
  const index = turn % names.length;
  const performer = names[index];
  status.textContent = replyModel ? performer + " is writing…" : "Loading the battle models…";
  try {
    if (!replyModel) {
      [replyModel, challengeModel] = await Promise.all([loadModel("context-reply"), loadModel("challenge")]);
    }
    const stats = find("model-stats");
    stats.textContent = "Reply model · " + replyModel.training.parameters.toLocaleString() +
      " trained parameters / Challenge model · " + challengeModel.training.parameters.toLocaleString() + " trained parameters";
    stats.hidden = false;
    // Give the browser a frame to announce the busy state before inference.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const earlier = performed[index];
    const reply = replyModel.generate({
      previous: previousChallenge,
      role: selectors[index].value,
      opponent: selectors[1 - index].value,
      cue,
      seed,
      earlier,
      temperature: samplingTemperature,
    });
    if (!reply) throw new Error("The model did not finish a two-line reply.");
    const challenge = challengeModel.generate({
      role: selectors[index].value,
      seed: seed ^ 0x9e3779b9,
      temperature: samplingTemperature,
      earlier: [...earlier, ...performed[1 - index], {text: reply, cue}],
    });
    if (!challenge) throw new Error("The model did not finish a two-line challenge.");
    appendVerse(performer, reply, challenge);
    earlier.push({text: reply, cue}, {text: challenge, cue: ""});
    previousChallenge = challenge;
    cue = "";
    turn++;
    reset.hidden = false;
    choices.replaceChildren();
    if (turn === totalTurns) {
      const heading = find("round");
      heading.textContent = "Battle complete";
      controls.hidden = true;
      heading.focus({preventScroll: true});
    } else {
      find("round").textContent = "Round " + (Math.floor(turn / names.length) + 1) + " of " + rounds;
      showChoices();
    }
  } catch {
    const loaded = Boolean(replyModel);
    if (turn === 0) {
      replyModel = challengeModel = undefined;
      matchup.disabled = false;
      find("model-stats").hidden = true;
    }
    status.textContent = loaded ? "The verse could not be generated. Try this turn again." : "Could not load the battle models. Check your connection and try again.";
    next.disabled = false;
    next.focus();
  } finally {
    busy = reset.disabled = temperature.disabled = false;
    for (const choice of choices.children) choice.disabled = false;
  }
};

reset.onclick = () => {
  turn = 0;
  replyModel = challengeModel = undefined;
  matchup.disabled = false;
  find("model-stats").hidden = true;
  previousChallenge = cue = "";
  for (const history of performed) history.length = 0;
  verses.replaceChildren();
  choices.replaceChildren();
  find("round").textContent = "Round 1 of 3";
  status.textContent = "Ready when you are.";
  next.textContent = "Start battle";
  controls.hidden = false;
  next.disabled = false;
  reset.hidden = true;
  next.focus();
};
