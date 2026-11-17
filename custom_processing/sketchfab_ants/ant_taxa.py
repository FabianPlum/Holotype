"""Ant genera, subfamilies and display names for the tunosemi Sketchfab models.

One table (SUBFAMILY_GENERA) serves both the selection of ant models (only genera listed here are
downloaded; tunosemi also uploads beetles, wasps, spiders and other outgroups) and the subfamily labels
of the fit report. Subfamilies follow the current AntCat classification as of 2026; Heteroponerinae for
Acanthoponera and Alfaria is the least certain assignment.
"""

import re

SUBFAMILY_GENERA = {
    "Myrmicinae": "Acanthomyrmex Acromyrmex Ancyridris Atopomyrmex Atta Basiceros Blepharidatta Carebara Cataulacus "
                  "Cephalotes Colobostruma Crematogaster Cyphomyrmex Dacetinops Daceton Dilobocondyla Eutetramorium "
                  "Hylomyrma Kalathomyrmex Manica Mayriella Melissotarsus Meranoplus Messor Metapone Microdaceton "
                  "Mycetomoellerius Mycocepurus Myrmica Orectognathus Pheidole Pristomyrmex Proatta Procryptocerus "
                  "Protalaridris Sericomyrmex Strumigenys Temnothorax Terataner Tetramorium Vollenhovia",
    "Ponerinae": "Anochetus Belonopelta Brachyponera Buniapone Diacamma Dinoponera Ectomomyrmex Hagensia Harpegnathos "
                 "Iroponera Leptogenys Megaponera Mesoponera Neoponera Odontomachus Odontoponera Ophthalmopone "
                 "Pachycondyla Paltothyreus Platythyrea Plectroctena Streblognathus Thaumatomyrmex",
    "Dorylinae": "Acanthostichus Aenictus Cylindromyrmex Dorylus Eciton Labidus Lioponera Neivamyrmex Nomamyrmex "
                 "Ooceraea Parasyscia Simopone Zasphinctus",
    "Formicinae": "Camponotus Colobopsis Dinomyrmex Echinopla Formica Gesomyrmex Gigantiops Myrmoteras Notoncus "
                  "Polyrhachis Santschiella",
    "Amblyoponinae": "Adetomyrma Amblyopone Apomyrma Fulakora Myopopone Mystrium Onychomyrmex Stigmatomma",
    "Ectatomminae": "Ectatomma Gnamptogenys Holcoponera Poneracantha Rhytidoponera Stictoponera Typhlomyrmex",
    "Dolichoderinae": "Dolichoderus Leptomyrmex Tapinoma",
    "Heteroponerinae": "Acanthoponera Alfaria",
    "Proceratiinae": "Discothyrea Proceratium",
    "Agroecomyrmecinae": "Ankylomyrma Tatuidris",
    "Pseudomyrmecinae": "Tetraponera",
    "Myrmeciinae": "Myrmecia Nothomyrmecia",
    "Paraponerinae": "Paraponera",
    "Leptanillinae": "Protanilla",
    "Martialinae": "Martialis",
    "Cretaceous stem ants †": "Gerontoformica Haidomyrmex Zigrasimecia",
}
GENUS_TO_SUBFAMILY = {g: s for s, gs in SUBFAMILY_GENERA.items() for g in gs.split()}

# Misspellings in tunosemi's Sketchfab titles -> valid genus
GENUS_ALIASES = {"Ection": "Eciton", "Pristmyrmex": "Pristomyrmex"}
# Word fixes for display names (species epithets and caste words)
WORD_FIXES = {"japonicm": "japonicum", "versicolorl": "versicolor", "miner": "minor",
              "Caldwell": "caldwelli", "Caldwelli": "caldwelli"}


def _words(model_name):
    name = model_name.replace("Ｍ", "M")  # full-width M in one title ("Ｍyrmoteras")
    return [w for w in re.split(r"[\s_.\-]+", name.strip()) if w]


def genus_of(model_name):
    """Valid genus name of a Sketchfab model title, or None if the first word is not an ant genus."""
    words = _words(model_name)
    if not words:
        return None
    g = words[0].capitalize()
    g = GENUS_ALIASES.get(g, g)
    return g if g in GENUS_TO_SUBFAMILY else None


def is_ant(model_name):
    return genus_of(model_name) is not None


def subfamily_of(model_name):
    g = genus_of(model_name)
    return GENUS_TO_SUBFAMILY.get(g) if g else None


def display_name(model_name):
    """Tidy taxon name from a Sketchfab title, e.g. 'pheidole Caldwelli miner' -> 'Pheidole caldwelli minor'."""
    words = [WORD_FIXES.get(w, w) for w in _words(model_name)]
    out = [GENUS_ALIASES.get(words[0].capitalize(), words[0].capitalize())] + [w.lower() for w in words[1:]]
    return " ".join({"sp": "sp.", "cf": "cf."}.get(w, w) for w in out)
