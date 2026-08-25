from pathlib import Path
import re

src = Path("intersection.net.xml")
dst = Path("intersection_v7.net.xml")

text = src.read_text(encoding="utf-8")

new_tl = """    <tlLogic id="center" type="static" programID="v7" offset="0">
        <!-- NS through + right -->
        <phase duration="20" state="GGrrrrrrGGrrrrrr"/>
        <phase duration="3"  state="yyrrrrrryyrrrrrr"/>

        <!-- NS protected left -->
        <phase duration="10" state="rrGrrrrrrrGrrrrr"/>
        <phase duration="3"  state="rryrrrrrrryrrrrr"/>

        <!-- EW through + right -->
        <phase duration="20" state="rrrrGGrrrrrrGGrr"/>
        <phase duration="3"  state="rrrryyrrrrrryyrr"/>

        <!-- EW protected left -->
        <phase duration="10" state="rrrrrrGrrrrrrrGr"/>
        <phase duration="3"  state="rrrrrryrrrrrrryr"/>
    </tlLogic>"""

pattern = re.compile(
    r'<tlLogic\s+id="center".*?</tlLogic>',
    re.DOTALL,
)

match = pattern.search(text)
if not match:
    raise SystemExit("Could not find the existing center <tlLogic> block.")

updated = text[:match.start()] + new_tl + text[match.end():]
dst.write_text(updated, encoding="utf-8")

print(f"Created {dst}")
print("Only the existing center <tlLogic> block was replaced.")
