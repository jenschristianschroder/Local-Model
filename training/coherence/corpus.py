"""Fresh AI-authored training text. No downloads or previous text inputs.

Templates are used only to construct offline training examples. Split by the
actor/companion combination so validation contains unseen combinations.
"""
import hashlib
import json
from pathlib import Path

ACTORS = ['Mira', 'Tobin', 'Nella', 'Orin', 'Bram', 'Luma', 'Fenn', 'Vera']
COMPANIONS = ['badger', 'goose', 'otter', 'mole', 'rabbit', 'penguin']
OBJECTS = ['silver bell', 'green mitten', 'wooden spoon', 'striped scarf']

# Each plot keeps a problem alive across four paragraphs and resolves it.
PLOTS = [
    [
        'On Christmas Eve, {a} needed the {o} for the village celebration. A {b} had locked it inside a cupboard. The cupboard would only open after hearing a joke about itself. {a} promised to bring the {o} back before dinner.',
        '{a} told the cupboard that its shelves looked tired. The cupboard did not laugh. The {b} explained that furniture preferred jokes about doors. Together they wrote a joke on a piece of wrapping paper, but the paper folded itself into a hat.',
        'The {b} wore the paper hat while {a} performed the joke. This time the cupboard laughed so hard that its door swung open. Inside, the {o} was resting on a cushion. {a} picked it up and thanked the cupboard for being a good audience.',
        '{a} carried the {o} back to the celebration, with the {b} following in the hat. Dinner could finally begin. They saved a bowl of soup for the cupboard. It asked them to pour the soup into a drawer, then complained that nobody had brought a spoon.'
    ],
    [
        '{a} was decorating the Christmas tree when a {b} borrowed the {o}. The {b} promised to return it before the lights came on. Unfortunately, a gust of wind carried the {o} onto the roof. {a} and the {b} needed a way to reach it.',
        'They fetched a ladder, but the ladder was afraid of heights. {a} held its lowest rung while the {b} offered encouraging words. The ladder climbed one step up itself and immediately asked for a holiday. The {o} was still on the roof.',
        '{a} suggested laying the ladder across a snowdrift. From there, the {b} could reach the roof without looking down. The plan worked. The {b} caught the {o} just as another gust of wind arrived and carefully carried it back to {a}.',
        'With the {o} safely returned, {a} finished decorating the tree. The {b} switched on the lights and apologized for borrowing things on windy days. They wrapped the ladder in a blanket. It spent Christmas lying flat and telling everyone about its mountain expedition.'
    ],
    [
        '{a} packed the {o} in a box for a Christmas delivery. A {b} offered to pull the little delivery cart. They had to cross the bridge before sunset. Halfway there, the cart stopped because one wheel had decided to become a dinner plate.',
        'The {b} tried asking the wheel to turn, but it demanded potatoes. {a} placed a potato on the wheel. This made the cart lean sideways, and the box almost fell into the snow. {a} caught the box while the {b} held the cart steady.',
        '{a} explained that a good dinner plate first had to reach the dinner table. The wheel agreed to finish the journey on that condition. The {b} pulled the cart across the bridge, and {a} kept the box containing the {o} balanced between both hands.',
        'They delivered the {o} before sunset. {a} opened the box for the waiting family, and the {b} parked the cart by the door. As promised, the wheel joined them at the table. It enjoyed the potatoes so much that it rolled home with gravy on its spokes.'
    ],
    [
        'At the Christmas market, {a} lost the {o}. A {b} offered to help find it before the stalls closed. They remembered leaving it beside a basket of apples. When they returned, the basket had grown legs and walked away.',
        '{a} followed tiny footprints through the snow. The {b} found an apple under a bench, then another near the bakery. They asked the baker whether a walking basket had passed. The baker pointed toward the music stall, where someone was loudly requesting a dance.',
        'The basket was dancing beside the music stall with the {o} tucked inside. {a} asked for it back. The basket wanted a partner for one last dance, so the {b} took its handle and spun it gently. After the dance, the basket returned the {o}.',
        '{a} put the {o} safely in a pocket and thanked the {b}. They reached home before the market closed. Later, the basket arrived with apples for Christmas dinner. It stayed beside the table, tapping its little feet whenever somebody rattled a fork.'
    ],
    [
        '{a} wanted to use the {o} in the Christmas play. A {b} was in charge of the stage. Just before the performance, a cloud floated indoors and hid the {o} in a thick mist. The audience was waiting, so {a} needed to clear the stage.',
        'The {b} opened a window, but the cloud refused to leave because it had bought a ticket. {a} offered it a seat at the back. The cloud tried sitting down and soaked three cushions. Somewhere in the mist, the {o} made a small clattering sound.',
        '{a} asked the cloud to play the winter sky above the stage. Pleased to be included, it rose toward the ceiling. The {b} found the {o} beneath a chair and handed it to {a}. With the stage clear, they began the Christmas play.',
        '{a} finished the performance with the {o} held high. The {b} bowed, and the cloud sprinkled a little snow for the final scene. The audience clapped. Afterward, the cloud asked for its own dressing room, preferably one with a waterproof floor.'
    ],
    [
        '{a} and a {b} were preparing Christmas dinner when the {o} vanished. A trail of flour led to the pantry. Inside, a loaf of bread was using the {o} as a royal treasure. It refused to give it back until somebody recognized its kingdom.',
        '{a} tried explaining that dinner was getting cold. The loaf declared that cold dinners were against the law. The {b} made a tiny crown from a paper napkin and asked what a bread kingdom needed. The loaf admitted that it had no subjects.',
        '{a} arranged six rolls around the loaf. The {b} announced that the rolls were loyal citizens who urgently needed their ruler at the table. Satisfied, the loaf returned the {o}. {a} carried it back to the kitchen and finished preparing dinner.',
        'The {b} placed the crowned loaf in the middle of the Christmas table. {a} used the recovered {o} and invited everyone to sit down. The bread kingdom enjoyed a peaceful evening. Its only complaint was that the butter kept sliding across the border.'
    ],
]


def build():
    splits = {'train': [], 'valid': [], 'test': []}
    for actor_index, actor in enumerate(ACTORS):
        for companion_index, companion in enumerate(COMPANIONS):
            fold = (actor_index + companion_index) % 8
            split = 'test' if fold == 0 else 'valid' if fold == 1 else 'train'
            for object_name in OBJECTS:
                for plot_index, plot in enumerate(PLOTS):
                    paragraphs = [p.format(a=actor, b=companion, o=object_name) for p in plot]
                    splits[split].append({'actor': actor, 'companion': companion,
                                          'object': object_name, 'plot': plot_index,
                                          'paragraphs': paragraphs})
    return splits


if __name__ == '__main__':
    root = Path(__file__).parent
    manifest = {'origin': 'AI-authored in this task; six new plots with synthetic substitutions',
                'externalTextInputs': [], 'split': 'Disjoint actor/companion pairs; plots shared',
                'limitations': 'Narrow templated corpus; tests measure recombination of known plots.',
                'splits': {}}
    for name, stories in build().items():
        data = json.dumps(stories, ensure_ascii=False, indent=2) + '\n'
        (root / f'{name}.json').write_text(data)
        manifest['splits'][name] = {'stories': len(stories), 'sha256': hashlib.sha256(data.encode()).hexdigest()}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
