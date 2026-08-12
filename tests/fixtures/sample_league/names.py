"""Name pools for the fictional sample dataset.

Deliberately invented names. The sample data exists so a new user can try the app
before they have their own league file, and it must never be mistaken for real
NFL data — so no real player's name appears here, and the generator combines
these lists rather than drawing from any roster.

NFL *team* abbreviations elsewhere in the sample data are real, because the
engine's stack and handcuff features are about teammates and would be untestable
against invented team codes. A real team code attached to an invented player
cannot be mistaken for a real player.
"""

from __future__ import annotations

FIRST_NAMES: tuple[str, ...] = (
    "Aaron", "Abel", "Adrian", "Alec", "Amari", "Anders", "Andre", "Angelo",
    "Anton", "Arlo", "Asher", "Aurelio", "Barrett", "Beau", "Bennett", "Blaise",
    "Bram", "Brant", "Brody", "Bruno", "Cadel", "Caius", "Calder", "Callum",
    "Camden", "Carlisle", "Casimir", "Cedric", "Cato", "Chance", "Cormac",
    "Crosby", "Dashiell", "Davion", "Deacon", "Declan", "Demetri", "Desmond",
    "Devlin", "Dorian", "Drexel", "Duncan", "Eamon", "Edison", "Elias",
    "Ellery", "Emory", "Enzo", "Everett", "Ezio", "Fabian", "Ferris", "Finnian",
    "Fletcher", "Ford", "Franco", "Gaspard", "Gideon", "Grady", "Griffin",
    "Halden", "Hollis", "Horace", "Huxley", "Ignatius", "Ilan", "Isaias",
    "Jarek", "Jarrett", "Jasper", "Jericho", "Joaquin", "Jonas", "Judah",
    "Kaeden", "Kalman", "Keanu", "Kelvin", "Kendrick", "Kieran", "Knox",
    "Lachlan", "Lamont", "Landry", "Lazarus", "Leland", "Lennox", "Linus",
    "Lorcan", "Lucian", "Maddox", "Magnus", "Malachi", "Marcellus", "Mateo",
    "Maverick", "Merrick", "Micah", "Milo", "Montrell", "Nash", "Nikolai",
    "Oberon", "Octavio", "Odell", "Orion", "Osric", "Paxton", "Percival",
    "Phineas", "Quill", "Quinton", "Ramsey", "Reeve", "Remy", "Ridley",
    "Roman", "Rowan", "Rudyard", "Sable", "Salem", "Santiago", "Sawyer",
    "Sebastien", "Shepard", "Silas", "Soren", "Stellan", "Sylvan", "Tavian",
    "Tennyson", "Thaddeus", "Theo", "Torin", "Tristram", "Ulysses", "Valen",
    "Vance", "Vaughn", "Verity", "Wendell", "Whitaker", "Wilder", "Xavien",
    "Yannick", "Zadok", "Zephyr",
)

LAST_NAMES: tuple[str, ...] = (
    "Abernathy", "Ackerley", "Adeyemi", "Alderton", "Ansley", "Applewhite",
    "Ashgrove", "Auclair", "Baldivar", "Ballantyne", "Barrowman", "Belmonte",
    "Berrigan", "Blackwood", "Bonnard", "Braddock", "Bramhall", "Brightwater",
    "Brockway", "Buchholz", "Cadogan", "Calloway", "Carbonell", "Cartwright",
    "Cavanagh", "Chesterton", "Clanton", "Coldwell", "Corriveau", "Crenshaw",
    "Cullimore", "Dalrymple", "Danforth", "Delacroix", "Demarais", "Dorsett",
    "Draycott", "Dunwoody", "Eastcott", "Ellingham", "Emberly", "Esposito",
    "Fairbourne", "Falconbridge", "Farnsworth", "Feathergill", "Fennimore",
    "Fitzroy", "Follansbee", "Fontaine", "Frostvale", "Gainsborough",
    "Galbraith", "Garrigan", "Ghiorso", "Gilliland", "Glendower", "Goodhew",
    "Grantham", "Greaveson", "Hallowell", "Hargreaves", "Hathersage",
    "Havelock", "Hawksley", "Heatherton", "Hollingsworth", "Hornbeck",
    "Huntington", "Ilderton", "Isenhour", "Jarnigan", "Jellicoe", "Kaltenbach",
    "Kearsley", "Kilbride", "Kingsbury", "Knightley", "Ladbroke", "Lamontagne",
    "Langstrom", "Lattimore", "Leventhal", "Lindqvist", "Littlewood",
    "Lovegrove", "Macalister", "Mainwaring", "Marchetti", "Mayfield",
    "Meriwether", "Mirandola", "Monckton", "Montcalm", "Mortlake", "Nethercott",
    "Newcombe", "Northrup", "Oakhurst", "Ollivander", "Orbison", "Ottoway",
    "Pemberton", "Penhaligon", "Pettigrew", "Pickersgill", "Plumtree",
    "Prendergast", "Quartermain", "Quennell", "Radcliffe", "Ravensworth",
    "Redgrave", "Rennenkampf", "Ridgeway", "Rothesay", "Rutherglen",
    "Sablewood", "Saltonstall", "Sandringham", "Scarborough", "Sedgewick",
    "Shackleton", "Sharplin", "Sheringham", "Silvestri", "Somerville",
    "Stanhope", "Sterling", "Stonebridge", "Strathmore", "Swinburne",
    "Tarkington", "Thackeray", "Thistlewood", "Throckmorton", "Tolliver",
    "Trevelyan", "Underhill", "Vandermolen", "Vasconcelos", "Vellano",
    "Verrazano", "Wadsworth", "Wainwright", "Walcott", "Wetherby",
    "Whitlingham", "Wickersham", "Winterbourne", "Wolstenholme", "Wyndham",
    "Yarborough", "Zabriskie",
)

MANAGER_NAMES: tuple[str, ...] = (
    "Alicia Brandt", "Dev Raghunathan", "Marcus Feld", "Priya Kaur",
    "Owen Castellanos", "Nadia Oyelaran", "Sam Whitlock", "Tomas Iversen",
    "Grace Lindqvist", "Bishop Adeyemi", "Renata Cardozo", "Kyle Mahoney",
)
"""The twelve fictional league members. Order is the default draft order."""

__all__ = ["FIRST_NAMES", "LAST_NAMES", "MANAGER_NAMES"]
