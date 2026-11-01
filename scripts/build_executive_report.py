"""Build a self-contained analyst-style DOCX using only the Python standard library."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "Real_Estate_Investment_Advisor_Executive_Report.docx"
MEDIA = {
    "cover": ROOT / "reports" / "executive_cover.png",
    "city": ROOT / "reports" / "eda" / "city_price_trends.png",
    "corr": ROOT / "reports" / "eda" / "correlation_heatmap.png",
}
for file in MEDIA.values():
    if not file.exists():
        raise FileNotFoundError(file)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
ET.register_namespace("w", W); ET.register_namespace("r", R)
ET.register_namespace("wp", WP); ET.register_namespace("a", A); ET.register_namespace("pic", PIC)
def tag(ns, name): return f"{{{ns}}}{name}"

doc = ET.Element(tag(W, "document"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
body = ET.SubElement(doc, tag(W, "body"))
rid = {"cover": "rId1", "city": "rId2", "corr": "rId3"}

def run(p, text, bold=False, color=None, size=None, italic=False):
    r = ET.SubElement(p, tag(W, "r")); rp = ET.SubElement(r, tag(W, "rPr"))
    if bold: ET.SubElement(rp, tag(W, "b"))
    if italic: ET.SubElement(rp, tag(W, "i"))
    if color: ET.SubElement(rp, tag(W, "color"), {tag(W, "val"): color})
    if size: ET.SubElement(rp, tag(W, "sz"), {tag(W, "val"): str(size)})
    t = ET.SubElement(r, tag(W, "t")); t.text = text

def para(text="", style=None, bold=False, color=None, size=None, italic=False, align=None, after=120):
    p = ET.SubElement(body, tag(W, "p")); pp = ET.SubElement(p, tag(W, "pPr"))
    if style: ET.SubElement(pp, tag(W, "pStyle"), {tag(W, "val"): style})
    spacing = ET.SubElement(pp, tag(W, "spacing"), {tag(W, "after"): str(after)})
    if align: ET.SubElement(pp, tag(W, "jc"), {tag(W, "val"): align})
    run(p, text, bold, color, size, italic)
    return p

def heading(text, level=1): para(text, f"Heading{level}", after=180)

def page_break():
    p = ET.SubElement(body, tag(W, "p")); r = ET.SubElement(p, tag(W, "r")); ET.SubElement(r, tag(W, "br"), {tag(W, "type"): "page"})

def picture(key, width_in, height_in):
    p = ET.SubElement(body, tag(W, "p")); ppr = ET.SubElement(p, tag(W, "pPr")); ET.SubElement(ppr, tag(W, "jc"), {tag(W, "val"): "center"})
    r = ET.SubElement(p, tag(W, "r")); drawing = ET.SubElement(r, tag(W, "drawing")); inline = ET.SubElement(drawing, tag(WP, "inline"), {"distT":"0","distB":"0","distL":"0","distR":"0"})
    cx, cy = int(width_in*914400), int(height_in*914400)
    ET.SubElement(inline, tag(WP,"extent"), {"cx":str(cx),"cy":str(cy)})
    ET.SubElement(inline, tag(WP,"docPr"), {"id":str(1+list(rid).index(key)),"name":key+" image"})
    graphic = ET.SubElement(inline, tag(A,"graphic")); gd = ET.SubElement(graphic, tag(A,"graphicData"), {"uri":PIC})
    pic = ET.SubElement(gd, tag(PIC,"pic")); nv = ET.SubElement(pic, tag(PIC,"nvPicPr")); ET.SubElement(nv,tag(PIC,"cNvPr"),{"id":"0","name":key+".png"}); ET.SubElement(nv,tag(PIC,"cNvPicPr"))
    fill = ET.SubElement(pic,tag(PIC,"blipFill")); ET.SubElement(fill,tag(A,"blip"),{tag(R,"embed"):rid[key]}); stretch=ET.SubElement(fill,tag(A,"stretch")); ET.SubElement(stretch,tag(A,"fillRect"))
    sp=ET.SubElement(pic,tag(PIC,"spPr")); xfrm=ET.SubElement(sp,tag(A,"xfrm")); ET.SubElement(xfrm,tag(A,"off"),{"x":"0","y":"0"}); ET.SubElement(xfrm,tag(A,"ext"),{"cx":str(cx),"cy":str(cy)}); geom=ET.SubElement(sp,tag(A,"prstGeom"),{"prst":"rect"}); ET.SubElement(geom,tag(A,"avLst"))

def table(rows, widths=None):
    tbl=ET.SubElement(body,tag(W,"tbl")); pr=ET.SubElement(tbl,tag(W,"tblPr")); borders=ET.SubElement(pr,tag(W,"tblBorders"))
    for edge in ("top","left","bottom","right","insideH","insideV"):
        ET.SubElement(borders,tag(W,edge),{tag(W,"val"):"single",tag(W,"sz"):"4",tag(W,"color"):"D8E1EB"})
    for ri,row in enumerate(rows):
        tr=ET.SubElement(tbl,tag(W,"tr"))
        for value in row:
            tc=ET.SubElement(tr,tag(W,"tc")); tcp=ET.SubElement(tc,tag(W,"tcPr"))
            if ri==0: ET.SubElement(tcp,tag(W,"shd"),{tag(W,"fill"):"17324D"})
            p=ET.SubElement(tc,tag(W,"p")); pp=ET.SubElement(p,tag(W,"pPr")); ET.SubElement(pp,tag(W,"spacing"),{tag(W,"after"):"60"})
            run(p,value,bold=(ri==0 or len(row)==2 and value.startswith("•")),color=("FFFFFF" if ri==0 else "17324D" if len(row)==2 and value.startswith("•") else None),size=19)

def bullet(text): para("•  "+text, "Normal", after=80)

# Cover
para("REAL ESTATE INVESTMENT ADVISOR", bold=True, color="17324D", size=20, after=80)
para("Executive analytics report", bold=True, color="17324D", size=34, after=120)
para("Indian housing listings • City-level price trends • Forecast validation", color="567086", size=17, after=240)
picture("cover", 6.25, 4.17)
para("Prepared from project data and model artifacts | 30 October 2026", color="567086", size=11, align="center", after=0)
page_break()

heading("Executive summary")
para("This project combines a 250,000-row Indian housing listing dataset with NHB RESIDEX city assessment-price indices. It supports listing exploration, descriptive comparisons, a transparent age-based screening proxy, and a five-year city-index scenario. The evidence is useful for analysis and product prototyping; it does not establish realized investment returns or property-specific resale accuracy.")
table([["Decision lens","What the evidence says"],
       ["Market trend scenario","The selected Theta model forecasts 20 quarters (five years) of city HPI. Cross-validation MAPE is 15.1%; the temporal holdout MAPE is 16.6% across 224 city-origin forecasts."],
       ["Investment screen","The “Good Investment” label is generated from assumed appreciation less an age adjustment. It is a rule demonstration, not an observed outcome."],
       ["Listing model","The classifier has holdout ROC AUC 0.501 and F1 0.293 against that proxy target. The scores do not support strong ranking skill."],
       ["Readiness","Suitable for exploratory decision support with clear caveats. Collect transaction-linked resale and operating-cost data before investment-grade deployment."]])
heading("At a glance",2)
table([["Listings","NHB HPI coverage","History","Forecast horizon"],["250,000","28 cities","51 quarters (Jun 2013–Dec 2025)","20 quarters to Dec 2030"]])
para("Interpretation: HPI forecast errors compare projected city indices with later index observations. They are not errors against individual property resales. Five-year evaluation windows overlap, so forecast errors are correlated.", italic=True, color="567086", size=10)

heading("What this analysis supports")
bullet("Compare asking-price distributions across available cities, property types, BHK bands, and listing attributes.")
bullet("Show a transparent screening rule whose assumptions can be changed and discussed.")
bullet("Translate projected city HPI movement into an illustrative price scenario when city coverage is available.")
bullet("Prioritize data collection and validation needed for a defensible property-level product.")
heading("What it does not support")
bullet("Claiming realized returns, investment success, or an independently validated five-year resale value.")
bullet("Treating asking prices as completed sale prices, or the city HPI as a property-level appraisal.")
bullet("Attributing price effects to crime or infrastructure: the listing source has no substantiated locality-level crime or infrastructure measures.")

page_break(); heading("Market context and analytical evidence")
para("The listing data is broad enough for descriptive segmentation, while the NHB index provides a separate city-level history for market movement. These sources have different units and observational bases. Keep listing-level distributions and city-level index forecasts distinct in dashboards and decisions.")
picture("city", 6.25, 3.45)
para("Figure 1. City asking-price trends in the supplied listing dataset. These are asking-price observations, not matched transaction histories.", italic=True, color="567086", size=9)
heading("Forecast evaluation",2)
table([["Evaluation stage","Window / sample","MAPE","Meaning"],
       ["Rolling-origin CV","12 origins; 336 city-origin forecasts","15.1%","Used to select the Theta method among eligible candidates."],
       ["Temporal holdout","8 origins; 224 city-origin forecasts","16.6%","Later historical origins, held out from selection; evaluates city HPI only."],
       ["Latest source index","Dec 2025","—","Forecast endpoint is Dec 2030; update the source series before operational use."]])
para("The selected Theta forecast improved holdout MAE by about 28.7% versus a last-value baseline (0.287 skill). This is historical index evidence; model performance can change across market regimes.")
heading("Proxy model diagnostics",2)
table([["Model output","Holdout result","Interpretation"],
       ["Investment classifier","Accuracy 50.7%; precision 20.8%; recall 49.4%; F1 29.3%; ROC AUC 0.501","Target is an age-driven rule label. ROC AUC near 0.5 indicates little ranking separation on this held-out city sample."],
       ["Proxy price regression","MAE ₹0.00073 lakh; R² ≈ 1.000","Target is formula-generated from listing price and assumptions. Near-perfect fit reflects the constructed target and is not evidence of resale accuracy."]])

page_break(); heading("Data, method, and governance")
heading("Sources and coverage",2)
table([["Source","Use in analysis","Coverage / caveat"],
       ["India housing prices CSV (Kaggle; project attribution)","Listing attributes, asking price, location, amenities","250,000 rows; no matched transaction dates or realized resale outcomes. Archive version and license need confirmation before redistribution."],
       ["NHB RESIDEX, HPI@Assessment Prices, Current-Q","City-level quarterly HPI trend and forecast","2,856 raw observations, 28 cities, 51 quarters. City benchmark, not individual property sales."],
       ["Previous-Q exports","Retained as raw provenance only","Values differ from Current-Q for many matched city-quarter records; views are not interchangeable and are not averaged."]])
heading("Preparation and modeling",2)
bullet("Cleaning removes exact duplicate rows, normalizes missing tokens, coerces numeric fields, excludes nonpositive price or area, and recomputes price per square foot.")
bullet("The classifier target uses 6% assumed annual appreciation, subtracts 0.2 percentage points per property year, and labels values above 4% as positive. Build year determines this target and is excluded from classifier predictors.")
bullet("Listing models use imputation, numeric scaling, and categorical one-hot encoding inside model pipelines. Classification is evaluated with city-grouped cross-validation and a city holdout.")
bullet("The HPI workflow compares candidate time-series methods with expanding-window, 20-quarter rolling-origin forecasts. Earlier origins select the model; the later temporal holdout scores the chosen approach.")
bullet("A city’s forecast HPI ratio is multiplied by a listing’s asking price. Listing capture dates and property-locality effects are unavailable, so alignment and proportional-growth assumptions remain unverified.")
heading("Data quality and provenance notes",2)
bullet("The listing manifest records zero exact duplicates, zero invalid-area/price removals, and no missing values in the modeled fields for the current input file.")
bullet("NHB files are HTML tables saved with an .xls extension. The merged output selects Current-Q consistently; the recorded merged-file SHA-256 is 93f72d2eb5e1554818ed091a9ece7a7feae0f9cfc820454279a8094fabe26319.")
bullet("Listing source hash: b0032196fcf336799daf45f7fed16199ff62cf23717e6dee76911467fb1d6367. Download timestamps and archive version are not embedded in the supplied source files.")
picture("corr", 5.7, 3.4)
para("Figure 2. Correlation heatmap for available listing fields. Correlation is descriptive and does not establish causation.", italic=True, color="567086", size=9)

page_break(); heading("Limitations and recommended next steps")
heading("Material limitations",2)
bullet("Outcome validity: no observed property purchase, sale, rent, maintenance, taxes, financing, or holding-period cash flows are provided.")
bullet("Target validity: classifier and regression targets are generated by project assumptions; high apparent fit cannot validate actual investment outcomes.")
bullet("Market transfer: the HPI series is city-wide and assessment-price based. It may not align to a listing’s asking-price date or reflect locality, condition, property type, or micro-market changes.")
bullet("Forecast uncertainty: history is limited, five-year windows overlap, and market regimes can shift. The empirical ratio band in project artifacts is indicative, not a calibrated prediction interval.")
bullet("Source maintenance: local HPI files end in Dec 2025 and are labeled base year 2018; the provenance notes say NHB now documents a newer linked base-year series. Reconcile series before deployment.")
heading("Recommended next steps",2)
table([["Priority","Action","Why it matters"],
       ["1 — Data","Acquire verified transaction records with property identifiers, dates, location, and sale prices; document source license and version.","Creates a real target for resale and return evaluation."],
       ["2 — Enrichment","Join locality-level rents, vacancy, taxes, maintenance, financing, commute/accessibility, and verified crime/infrastructure measures.","Enables net-return and neighborhood analysis with attributable evidence."],
       ["3 — Validation","Use time-based and geography-based holdouts; compare to simple baselines; report calibration, error by city/segment, and uncertainty.","Tests whether the model generalizes and where it fails."],
       ["4 — Operations","Refresh and reconcile NHB base-year links, preserve raw downloads, log hashes/date, and monitor drift and coverage.","Keeps inputs auditable and current."],
       ["5 — Product","Label outputs as asking-price comparisons and city-index scenarios; expose assumptions and coverage gaps in the interface.","Prevents false precision and supports informed review."]])
heading("Reproducibility",2)
para("Project commands: python scripts/run_eda.py; python scripts/merge_nhb_hpi.py; python scripts/train.py. Key documentation: docs/methodology.md and docs/data_provenance.md. Current forecast artifacts: artifacts/hpi_forecast_model.json and reports/hpi_forecast_metrics.json. Model outputs are specific to the recorded inputs and assumptions; regenerate reports after changing data or configuration.")
para("Analyst conclusion", "Heading2")
para("Treat this as an exploratory analytics prototype with a useful city-market scenario and a transparent but assumption-driven screening rule. It is not yet a validated investment recommendation system. Decision-grade claims require transaction-linked outcomes, net-return inputs, and forward validation on unseen periods and locations.", bold=True, color="17324D")

sect=ET.SubElement(body,tag(W,"sectPr")); pg=ET.SubElement(sect,tag(W,"pgSz"),{tag(W,"w"):"12240",tag(W,"h"):"15840"}); ET.SubElement(sect,tag(W,"pgMar"),{tag(W,"top"):"850",tag(W,"right"):"900",tag(W,"bottom"):"850",tag(W,"left"):"900",tag(W,"header"):"400",tag(W,"footer"):"400",tag(W,"gutter"):"0"})

styles=ET.Element(tag(W,"styles")); normal=ET.SubElement(styles,tag(W,"style"),{tag(W,"type"):"paragraph",tag(W,"default"):"1",tag(W,"styleId"):"Normal"}); np=ET.SubElement(normal,tag(W,"pPr")); ET.SubElement(np,tag(W,"spacing"),{tag(W,"line"):"276",tag(W,"after"):"100"}); nr=ET.SubElement(normal,tag(W,"rPr")); ET.SubElement(nr,tag(W,"rFonts"),{tag(W,"ascii"):"Aptos",tag(W,"hAnsi"):"Aptos"}); ET.SubElement(nr,tag(W,"sz"),{tag(W,"val"):"21"}); ET.SubElement(nr,tag(W,"color"),{tag(W,"val"):"263746"})
for lvl,sz in [(1,32),(2,25)]:
    st=ET.SubElement(styles,tag(W,"style"),{tag(W,"type"):"paragraph",tag(W,"styleId"):f"Heading{lvl}"}); ppr=ET.SubElement(st,tag(W,"pPr")); ET.SubElement(ppr,tag(W,"keepNext")); ET.SubElement(ppr,tag(W,"spacing"),{tag(W,"before"):"240",tag(W,"after"):"120"}); rpr=ET.SubElement(st,tag(W,"rPr")); ET.SubElement(rpr,tag(W,"b")); ET.SubElement(rpr,tag(W,"color"),{tag(W,"val"):"17324D"}); ET.SubElement(rpr,tag(W,"sz"),{tag(W,"val"):str(sz)})

rels=ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationships")
for key, r in rid.items(): ET.SubElement(rels,"{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",{"Id":r,"Type":"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image","Target":"media/"+key+".png"})
content_types=ET.Element("{http://schemas.openxmlformats.org/package/2006/content-types}Types")
ET.SubElement(content_types,"{http://schemas.openxmlformats.org/package/2006/content-types}Default",{"Extension":"rels","ContentType":"application/vnd.openxmlformats-package.relationships+xml"})
ET.SubElement(content_types,"{http://schemas.openxmlformats.org/package/2006/content-types}Default",{"Extension":"xml","ContentType":"application/xml"})
ET.SubElement(content_types,"{http://schemas.openxmlformats.org/package/2006/content-types}Default",{"Extension":"png","ContentType":"image/png"})
ET.SubElement(content_types,"{http://schemas.openxmlformats.org/package/2006/content-types}Override",{"PartName":"/word/document.xml","ContentType":"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"})
ET.SubElement(content_types,"{http://schemas.openxmlformats.org/package/2006/content-types}Override",{"PartName":"/word/styles.xml","ContentType":"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
with ZipFile(OUT,"w",ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml",ET.tostring(content_types,encoding="utf-8",xml_declaration=True))
    rootrels=ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationships")
    ET.SubElement(rootrels,"{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",{"Id":"rId1","Type":"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument","Target":"word/document.xml"})
    z.writestr("_rels/.rels",ET.tostring(rootrels,encoding="utf-8",xml_declaration=True))
    z.writestr("word/document.xml",ET.tostring(doc,encoding="utf-8",xml_declaration=True))
    z.writestr("word/_rels/document.xml.rels",ET.tostring(rels,encoding="utf-8",xml_declaration=True))
    z.writestr("word/styles.xml",ET.tostring(styles,encoding="utf-8",xml_declaration=True))
    for key,path in MEDIA.items(): z.write(path,"word/media/"+key+".png")
print(OUT)
