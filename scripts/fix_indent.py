import re

path = "app.py"
src = open(path, encoding="utf-8").read()

def dedent_region(text, start_anchor, end_anchor):
    i = text.find(start_anchor)
    j = text.find(end_anchor, i)
    if i == -1 or j == -1:
        print(f"  WARNING: anchor not found: {start_anchor[:60]!r}")
        return text
    region = text[i:j]
    fixed  = re.sub(r"^        ", "    ", region, flags=re.MULTILINE)
    return text[:i] + fixed + text[j:]

# excess_vs_nifty body
src = dedent_region(src,
    '        n_avail = summary["nifty_available"]',
    "\ndef _render_sector_rotation()")

# sector_rotation body
src = dedent_region(src,
    "        st.divider()\n        st.markdown(\"### Sector Rotation",
    "\ndef _render_mae_analysis()")

# mae body
src = dedent_region(src,
    '        valid_mae = mae_df["MAE % (worst intraday dip)"]',
    "\ndef tab_deep_insights()")

open(path, "w", encoding="utf-8").write(src)
print("Done")
