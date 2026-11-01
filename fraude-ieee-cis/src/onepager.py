"""One-pager PDF (une page A4) généré à partir des artefacts."""
from __future__ import annotations

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import models as MD
from . import store


def _money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


def build_pdf(author: str = "") -> bytes:
    meta, res, imp = store.meta(), store.results(), store.shap_importance()
    best = res["best_model"]
    xg = res["models"]["xgboost"]
    sup_best = max((k for k in res["models"] if MD.is_supervised(k)),
                   key=lambda k: res["models"][k]["validation"]["pr_auc"])
    sb = res["models"][sup_best]
    opt = sb["optimal"]
    sp = meta["splits"]

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.4, leading=10.6)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10,
                       textColor=colors.HexColor("#8E2431"), spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19,
                           alignment=0, textColor=colors.HexColor("#1F2937"))
    small = ParagraphStyle("s", parent=body, fontSize=7.2, leading=9, textColor=colors.grey)

    src_note = ("DONNEES SYNTHETIQUES (demo) - " if meta["source"] == "synthetic" else "")
    story = [Paragraph("Detection de fraude explicable - IEEE-CIS", title),
             Paragraph(f"{src_note}Data Mining, ENSI 2026-2027 - {author + ' - ' if author else ''}{date.today():%d/%m/%Y}", small),
             Paragraph("1. Probleme metier", h),
             Paragraph("Une equipe d'analystes ne peut examiner qu'un nombre limite d'alertes par jour. L'outil classe "
                       "les transactions par risque, fixe le seuil d'alerte en minimisant un cout en montants "
                       "(fraudes manquees + cout d'analyse des alertes) et explique chaque alerte.", body),
             Paragraph("2. Donnees et methode", h),
             Paragraph(f"{meta['n_rows']:,} transactions e-commerce (Vesta), {meta['fraud_rate']:.2f} % de fraudes, "
                       f"{len(meta['features'])} variables dont 4 comportementales calculees sur le passe uniquement "
                       f"(volume sur 1 h et 24 h, ecart au profil 30 j, delai depuis l'achat precedent) sur une identite "
                       f"reconstituee (approximation). Decoupage chronologique 60/20/20 : train {sp['train']['rows']:,}, "
                       f"validation {sp['validation']['rows']:,}, test {sp['test']['rows']:,}. Strategie contre le desequilibre "
                       f"retenue : <b>{meta['selected_strategy']}</b> (comparee en validation, appliquee au train seul). "
                       f"Budget d'alertes k = {meta['budget_frac']:.0%} des transactions.".replace(",", " "), body),
             Paragraph("3. Meilleur modele et resultats (jeu de test)", h),
             Paragraph(f"Meilleur modele supervise : <b>{MD.label(sup_best)}</b> (PR-AUC validation {sb['validation']['pr_auc']:.3f}). "
                       f"Meilleur modele non supervise : Isolation Forest. Cout d'analyse par alerte suppose : "
                       f"{res['cost_per_alert']:.0f} $ (hypothese modifiable).", body)]

    rows = [["Modele", "Type", "PR-AUC", "ROC-AUC", f"Prec.@k", f"Rappel@k"]]
    for k, v in res["models"].items():
        t = v["test"]
        rows.append([v["label"], v["kind"], f"{t['pr_auc']:.3f}", f"{t['roc_auc']:.3f}",
                     f"{t['precision_at_k']:.2f}", f"{t['recall_at_k']:.2f}"])
    tb = Table(rows, colWidths=[4.2 * cm, 2.8 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm])
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.6), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3E6E8")),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story += [Spacer(1, 3), tb, Spacer(1, 3)]
    o = opt["test"]
    story.append(Paragraph(
        f"Seuil choisi sur la validation (cout minimal) puis applique au test : {o['n_alerts']:,} alertes, "
        f"precision {o['precision']:.2f}, rappel {o['recall']:.2f}, F2 {o['f2']:.2f}, MCC {o['mcc']:.2f}. "
        f"Cout {_money(o['cost'])} contre {_money(o['no_model_cost'])} sans modele, soit une economie de "
        f"<b>{_money(o['saving'])}</b> ({100 * o['saving'] / max(o['no_model_cost'], 1):.0f} %).".replace(",", " "), body))

    top = ", ".join(f"{i['label']}" for i in imp[:5])
    story += [Paragraph("4. Explications cles (SHAP)", h),
              Paragraph(f"Variables les plus influentes : {top}. Chaque alerte est accompagnee de 3 a 5 codes motifs "
                        f"(variables qui poussent le plus le score vers la fraude) via l'API FastAPI et un graphique waterfall.", body),
              Paragraph("5. Recommandations et limites", h),
              Paragraph("<b>Recommandations</b> : deployer le modele supervise avec le seuil optimal en cout, regler le budget k "
                        "selon la capacite reelle des analystes, ajouter l'Isolation Forest comme filet de securite pour les schemas inedits, "
                        "re-entrainer regulierement (derive des schemas de fraude) et suivre le taux de fausses alertes.", body),
              Paragraph("<b>Limites</b> : identite client reconstituee de facon approximative ; variables Vesta anonymisees (explications "
                        "limitees) ; etiquettes tardives (fraudes non detectees comptees comme normales) ; cout d'alerte fixe suppose ; "
                        "methodes non supervisees nettement moins performantes (inhabituel n'est pas frauduleux) ; "
                        "resultats valables pour cette periode et a reverifier avant toute mise en production.", body)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm, title="One-pager fraude IEEE-CIS")
    w, hgt = A4[0] - 3 * cm, A4[1] - 2.5 * cm
    doc.build([KeepInFrame(w, hgt, story, mode="shrink")])   # garantit une seule page
    return buf.getvalue()
