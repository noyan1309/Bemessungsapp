# =============================================================================
# STREAMLIT-APP: BEMESSUNG EINES EINFELDTRÄGERS
# =============================================================================

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ReportLab für den PDF-Bericht
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =============================================================================
# 1. SEITENEINSTELLUNGEN
# =============================================================================

st.set_page_config(
    page_title="Bemessung Einfeldträger",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# 2. BAUTABELLEN UND DATENBANKEN
# =============================================================================

PROFIL_DATENBANK = {
    "IPE 80":  {"h": 80,  "g": 6.0,   "Wel": 20.0},
    "IPE 100": {"h": 100, "g": 8.1,   "Wel": 34.2},
    "IPE 120": {"h": 120, "g": 10.4,  "Wel": 54.7},
    "IPE 140": {"h": 140, "g": 12.9,  "Wel": 77.3},
    "IPE 160": {"h": 160, "g": 15.8,  "Wel": 109.0},
    "IPE 180": {"h": 180, "g": 18.8,  "Wel": 146.0},
    "IPE 200": {"h": 200, "g": 22.4,  "Wel": 194.0},
    "IPE 220": {"h": 220, "g": 26.2,  "Wel": 252.0},
    "IPE 240": {"h": 240, "g": 30.7,  "Wel": 324.0},
    "IPE 270": {"h": 270, "g": 36.1,  "Wel": 429.0},
    "IPE 300": {"h": 300, "g": 42.2,  "Wel": 557.0},
    "IPE 330": {"h": 330, "g": 49.1,  "Wel": 713.0},
    "IPE 360": {"h": 360, "g": 57.1,  "Wel": 904.0},
    "IPE 400": {"h": 400, "g": 66.3,  "Wel": 1160.0},
    "IPE 450": {"h": 450, "g": 77.6,  "Wel": 1500.0},
    "IPE 500": {"h": 500, "g": 90.7,  "Wel": 1930.0},
    "IPE 550": {"h": 550, "g": 106.0, "Wel": 2440.0},
    "IPE 600": {"h": 600, "g": 122.0, "Wel": 3070.0},
}

STAHLGUETEN = {
    "S235": 235.0,
    "S355": 355.0,
    "S460": 460.0,
}


# =============================================================================
# 3. STATIK-ALGORITHMUS
# =============================================================================

@st.cache_data(show_spinner=False)
def berechne_statik(L, delta_g_k, Q_k, profil_name):
    """
    Berechnet die maßgebenden Schnittgrößen für ein IPE-Profil.

    Eingaben:
    L            = Spannweite [m]
    delta_g_k    = maximale Dreieckslast [kN/m]
    Q_k          = wandernde Einzellast [kN]
    profil_name  = Name des untersuchten IPE-Profils

    Rückgabe:
    Maximales Moment, maximale Querkraft, maßgebende Laststellung,
    x-Koordinaten sowie Momenten- und Querkraftverlauf.
    """

    # Teilsicherheitsbeiwerte
    gamma_G = 1.35
    gamma_Q = 1.50

    # Bemessungswert der Dreieckslast
    q_delta_d = delta_g_k * gamma_G

    # Eigengewicht des ausgewählten IPE-Profils
    profilgewicht = PROFIL_DATENBANK[profil_name]["g"]      # kg/m
    g_k = (profilgewicht * 9.81) / 1000.0                  # kN/m
    q_g_d = g_k * gamma_G                                  # kN/m

    # Auflagerkräfte aus Dreieckslast und Eigengewicht
    A_g = (q_delta_d * L / 6.0) + (q_g_d * L / 2.0)
    B_g = (q_delta_d * L / 3.0) + (q_g_d * L / 2.0)

    # Bemessungswert der wandernden Einzellast
    Q_d = Q_k * gamma_Q

    # Diskretisierung des Trägers und der Laststellungen
    x_punkte = np.linspace(0.0, L, 601)

    # Schnittgrößen aus den ständigen Lasten
    M_g = (
        A_g * x_punkte
        - (q_delta_d * x_punkte**3) / (6.0 * L)
        - (q_g_d * x_punkte**2) / 2.0
    )

    V_g = (
        A_g
        - (q_delta_d * x_punkte**2) / (2.0 * L)
        - q_g_d * x_punkte
    )

    # Startwerte für die Suche nach den Maximalwerten
    M_Ed_max = 0.0
    V_Ed_max = 0.0
    massgebende_laststellung = 0.0

    # Die Einzellast wandert über 601 mögliche Positionen
    for x_Q in x_punkte:

        # Auflagerkräfte aus der Einzellast
        A_q = Q_d * (L - x_Q) / L
        B_q = Q_d * x_Q / L

        # Moment aus der Einzellast
        M_q = np.where(
            x_punkte <= x_Q,
            A_q * x_punkte,
            A_q * x_punkte - Q_d * (x_punkte - x_Q),
        )

        # Querkraft aus der Einzellast
        V_q = np.where(
            x_punkte <= x_Q,
            A_q,
            A_q - Q_d,
        )

        # Überlagerung aller gleichzeitig wirkenden Lasten
        M_aktuell = M_g + M_q
        V_aktuell = V_g + V_q

        # Größtes positives Biegemoment
        aktuelles_M_max = np.max(M_aktuell)

        if aktuelles_M_max > M_Ed_max:
            M_Ed_max = aktuelles_M_max
            massgebende_laststellung = x_Q

        # Größter Querkraftbetrag
        aktuelles_V_max = np.max(np.abs(V_aktuell))

        if aktuelles_V_max > V_Ed_max:
            V_Ed_max = aktuelles_V_max

    # Schnittgrößenverläufe für die momentmaßgebende Laststellung
    A_q_massgebend = Q_d * (L - massgebende_laststellung) / L

    M_q_massgebend = np.where(
        x_punkte <= massgebende_laststellung,
        A_q_massgebend * x_punkte,
        A_q_massgebend * x_punkte
        - Q_d * (x_punkte - massgebende_laststellung),
    )

    V_q_massgebend = np.where(
        x_punkte <= massgebende_laststellung,
        A_q_massgebend,
        A_q_massgebend - Q_d,
    )

    M_plot = M_g + M_q_massgebend
    V_plot = V_g + V_q_massgebend

    # Position des maximalen Momentes im Träger
    x_M_max = x_punkte[np.argmax(M_plot)]

    return {
        "M_Ed_max": M_Ed_max,
        "V_Ed_max": V_Ed_max,
        "laststellung": massgebende_laststellung,
        "x_punkte": x_punkte,
        "M_plot": M_plot,
        "V_plot": V_plot,
        "x_M_max": x_M_max,
        "A_g": A_g,
        "B_g": B_g,
        "q_g_d": q_g_d,
        "q_delta_d": q_delta_d,
        "Q_d": Q_d,
    }


# =============================================================================
# 4. ITERATIVE PROFILAUSWAHL
# =============================================================================

def bemesse_einfeldtraeger(L, delta_g_k, Q_k, stahlguete):
    """
    Prüft die IPE-Profile vom leichtesten zum schwersten und gibt
    das erste Profil zurück, das den Spannungsnachweis erfüllt.
    """

    f_y = STAHLGUETEN[stahlguete]

    # Umrechnung von N/mm² in kN/m²
    sigma_Rd = f_y * 1000.0

    sortierte_profile = sorted(
        PROFIL_DATENBANK.keys(),
        key=lambda name: PROFIL_DATENBANK[name]["h"],
    )

    for profil_name in sortierte_profile:

        statik = berechne_statik(
            L,
            delta_g_k,
            Q_k,
            profil_name,
        )

        # Elastisches Widerstandsmoment von cm³ in m³
        W_el_m3 = PROFIL_DATENBANK[profil_name]["Wel"] * 1e-6

        # Biegespannung in kN/m²
        sigma_Ed = statik["M_Ed_max"] / W_el_m3

        if sigma_Ed <= sigma_Rd:

            ausnutzung = (sigma_Ed / sigma_Rd) * 100.0

            return {
                "L": L,
                "delta_g_k": delta_g_k,
                "Q_k": Q_k,
                "stahlguete": stahlguete,
                "f_y": f_y,
                "profil": profil_name,
                "profilhoehe": PROFIL_DATENBANK[profil_name]["h"],
                "profilgewicht": PROFIL_DATENBANK[profil_name]["g"],
                "W_el": PROFIL_DATENBANK[profil_name]["Wel"],
                "sigma_Ed": sigma_Ed / 1000.0,
                "sigma_Rd": f_y,
                "ausnutzung": ausnutzung,
                **statik,
            }

    return None


# =============================================================================
# 5. DIAGRAMM ERSTELLEN
# =============================================================================

def erstelle_diagramm(ergebnis):
    """
    Erstellt Momenten- und Querkraftdiagramm.
    Positive Werte werden entsprechend der Aufgabenstellung nach unten dargestellt.
    """

    x_punkte = ergebnis["x_punkte"]
    M_plot = ergebnis["M_plot"]
    V_plot = ergebnis["V_plot"]
    laststellung = ergebnis["laststellung"]

    abbildung, (achse_m, achse_v) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
    )

    abbildung.suptitle(
        f"Schnittgrößenverläufe für {ergebnis['profil']}",
        fontsize=13,
        fontweight="bold",
    )

    # Momentenverlauf
    achse_m.plot(
        x_punkte,
        M_plot,
        color="crimson",
        linewidth=2,
        label="M_Ed",
    )

    achse_m.fill_between(
        x_punkte,
        M_plot,
        color="crimson",
        alpha=0.15,
    )

    achse_m.axvline(
        x=laststellung,
        color="black",
        linestyle="--",
        label=f"Einzellast bei x = {laststellung:.2f} m",
    )

    achse_m.scatter(
        ergebnis["x_M_max"],
        ergebnis["M_Ed_max"],
        color="black",
        zorder=5,
        label=f"M_Ed,max = {ergebnis['M_Ed_max']:.2f} kNm",
    )

    achse_m.set_ylabel("Biegemoment M [kNm]")
    achse_m.grid(True, linestyle=":")
    achse_m.invert_yaxis()
    achse_m.legend(loc="upper right", fontsize=8)

    # Querkraftverlauf
    achse_v.plot(
        x_punkte,
        V_plot,
        color="navy",
        linewidth=2,
        label="V_Ed",
    )

    achse_v.fill_between(
        x_punkte,
        V_plot,
        color="navy",
        alpha=0.15,
    )

    achse_v.axvline(
        x=laststellung,
        color="black",
        linestyle="--",
    )

    achse_v.set_xlabel("Trägerlänge x [m]")
    achse_v.set_ylabel("Querkraft V [kN]")
    achse_v.grid(True, linestyle=":")
    achse_v.invert_yaxis()

    plt.tight_layout()

    # Diagramm zusätzlich als PNG im Arbeitsspeicher sichern
    bild_speicher = io.BytesIO()

    abbildung.savefig(
        bild_speicher,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    bild_speicher.seek(0)

    return abbildung, bild_speicher.getvalue()


# =============================================================================
# 6. PDF-BERICHT ERSTELLEN
# =============================================================================

def erstelle_pdf(ergebnis, diagramm_png):
    """
    Erstellt den Ergebnisbericht vollständig im Arbeitsspeicher.
    Dadurch funktioniert der Download auch in Streamlit Community Cloud.
    """

    pdf_speicher = io.BytesIO()

    dokument = SimpleDocTemplate(
        pdf_speicher,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    inhalt = []
    styles = getSampleStyleSheet()

    titel_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )

    text_style = ParagraphStyle(
        "DocText",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
    )

    kopfzeilen_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    inhalt.append(Paragraph("<b>Ergebnisbericht</b>", titel_style))
    inhalt.append(Spacer(1, 10))

    ausnutzung_text = (
        f"<b><font color='green'>{ergebnis['ausnutzung']:.1f} % "
        f"(Nachweis erfüllt)</font></b>"
    )

    daten = [
        [
            Paragraph("Parameter", kopfzeilen_style),
            Paragraph("Ermittelter Wert", kopfzeilen_style),
        ],
        [
            Paragraph("Gewählte Stützweite L", text_style),
            Paragraph(f"{ergebnis['L']:.2f} m", text_style),
        ],
        [
            Paragraph("Ausbaulast Δg", text_style),
            Paragraph(f"{ergebnis['delta_g_k']:.2f} kN/m", text_style),
        ],
        [
            Paragraph("Einzellast Q", text_style),
            Paragraph(f"{ergebnis['Q_k']:.2f} kN", text_style),
        ],
        [
            Paragraph("Gewählte Stahlgüte", text_style),
            Paragraph(ergebnis["stahlguete"], text_style),
        ],
        [
            Paragraph("Wirtschaftlichstes IPE-Profil", text_style),
            Paragraph(ergebnis["profil"], text_style),
        ],
        [
            Paragraph("Maximales Design-Moment M_Ed", text_style),
            Paragraph(f"{ergebnis['M_Ed_max']:.2f} kNm", text_style),
        ],
        [
            Paragraph("Maximale Design-Querkraft V_Ed", text_style),
            Paragraph(f"{ergebnis['V_Ed_max']:.2f} kN", text_style),
        ],
        [
            Paragraph("Ungünstigste Laststelle", text_style),
            Paragraph(f"x = {ergebnis['laststellung']:.2f} m", text_style),
        ],
        [
            Paragraph("Vorhandene Biegespannung σ_Ed", text_style),
            Paragraph(f"{ergebnis['sigma_Ed']:.1f} N/mm²", text_style),
        ],
        [
            Paragraph("Zulässige Biegespannung σ_Rd", text_style),
            Paragraph(f"{ergebnis['sigma_Rd']:.1f} N/mm²", text_style),
        ],
        [
            Paragraph("Profilausnutzung", text_style),
            Paragraph(ausnutzung_text, text_style),
        ],
    ]

    tabelle = Table(
        daten,
        colWidths=[240, 180],
    )

    tabelle.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1A365D")),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    inhalt.append(tabelle)
    inhalt.append(Spacer(1, 15))

    inhalt.append(
        Paragraph(
            "<b>Visualisierung der maßgebenden Schnittgrößenverläufe:</b>",
            styles["Heading2"],
        )
    )

    inhalt.append(Spacer(1, 5))

    diagramm_speicher = io.BytesIO(diagramm_png)

    diagramm_bild = RLImage(
        diagramm_speicher,
        width=470,
        height=365,
    )

    inhalt.append(diagramm_bild)

    dokument.build(inhalt)

    pdf_speicher.seek(0)

    return pdf_speicher.getvalue()


# =============================================================================
# 7. STREAMLIT-BENUTZEROBERFLÄCHE
# =============================================================================

st.markdown(
    """
    <style>
        .app-kopf {
            background-color: #1A365D;
            color: white;
            padding: 18px 22px;
            border-radius: 12px;
            margin-bottom: 18px;
        }

        .app-kopf h1 {
            color: white;
            margin: 0;
            font-size: 28px;
        }

        .app-kopf p {
            margin: 6px 0 0 0;
            color: #E2E8F0;
        }

        .ergebnis-kasten {
            border: 1px solid #CBD5E1;
            background-color: #F8FAFC;
            padding: 18px;
            border-radius: 10px;
        }
    </style>

    <div class="app-kopf">
        <h1>📊 Bemessung eines Einfeldträgers</h1>
        <p>
            Ermittlung der ungünstigsten Laststellung und automatische
            Auswahl des wirtschaftlichsten IPE-Profils
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Eingabebereich
with st.container(border=True):

    st.subheader("Eingabewerte")

    spalte_1, spalte_2 = st.columns(2)

    with spalte_1:

L = st.number_input(
    "Spannweite L [m]",
    min_value=0.00,
    value=0.00,
    step=0.10,
    format="%.2f",
)

delta_g_k = st.number_input(
    "Ausbaulast (Δg)",
    min_value=0.00,
    value=0.00,
    step=0.10,
    format="%.2f",
)

Q_k = st.number_input(
    "Wandernde Einzellast Q [kN]",
    min_value=0.00,
    value=0.00,
    step=0.10,
    format="%.2f",
)

    with spalte_2:

        Q_k = st.number_input(
            "Einzellast Q [kN]",
            min_value=0.00,
            value=00.00,
            step=0.10,
            format="%.2f",
        )

        stahlguete = st.selectbox(
            "Stahlgüte",
            options=list(STAHLGUETEN.keys()),
            index=0,
        )

    berechnen = st.button(
        "🧮 Berechnung durchführen",
        type="primary",
        use_container_width=True,
    )


# =============================================================================
# 8. BERECHNUNG AUSFÜHREN
# =============================================================================

if berechnen:

    if L <= 0.0 or (delta_g_k <= 0.0 and Q_k <= 0.0):

        st.error(
            "Bitte geben Sie eine positive Spannweite und mindestens "
            "eine Last größer als 0 ein."
        )

    else:

        with st.spinner(
            "Laststellungen und IPE-Profile werden untersucht ..."
        ):

            ergebnis = bemesse_einfeldtraeger(
                L,
                delta_g_k,
                Q_k,
                stahlguete,
            )

        if ergebnis is None:

            st.session_state["ergebnis"] = None
            st.session_state["diagramm_png"] = None

            st.error(
                "Kein geeignetes Profil gefunden. "
            )

        else:

            abbildung, diagramm_png = erstelle_diagramm(ergebnis)

            st.session_state["ergebnis"] = ergebnis
            st.session_state["diagramm_png"] = diagramm_png

            plt.close(abbildung)


# =============================================================================
# 9. ERGEBNISSE ANZEIGEN
# =============================================================================

ergebnis = st.session_state.get("ergebnis")
diagramm_png = st.session_state.get("diagramm_png")

if ergebnis is not None and diagramm_png is not None:

    st.subheader("Ergebnis der Bemessung")

    kennzahl_1, kennzahl_2, kennzahl_3, kennzahl_4 = st.columns(4)

    kennzahl_1.metric(
        "Wirtschaftlichstes Profil",
        ergebnis["profil"],
    )

    kennzahl_2.metric(
        "Max. Moment M_Ed",
        f"{ergebnis['M_Ed_max']:.2f} kNm",
    )

    kennzahl_3.metric(
        "Max. Querkraft V_Ed",
        f"{ergebnis['V_Ed_max']:.2f} kN",
    )

    kennzahl_4.metric(
        "Profilausnutzung",
        f"{ergebnis['ausnutzung']:.1f} %",
    )

    st.markdown(
        f"""
        <div class="ergebnis-kasten">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td><b>Maßgebende Laststellung Q:</b></td>
                    <td style="text-align: right;">
                        x = {ergebnis['laststellung']:.2f} m
                    </td>
                </tr>
                <tr>
                    <td><b>Position des maximalen Moments:</b></td>
                    <td style="text-align: right;">
                        x = {ergebnis['x_M_max']:.2f} m
                    </td>
                </tr>
                <tr>
                    <td><b>Spannungsnachweis:</b></td>
                    <td style="text-align: right;">
                        {ergebnis['sigma_Ed']:.1f} N/mm²
                        ≤ {ergebnis['sigma_Rd']:.0f} N/mm²
                    </td>
                </tr>
                <tr>
                    <td><b>Profilgewicht:</b></td>
                    <td style="text-align: right;">
                        {ergebnis['profilgewicht']:.1f} kg/m
                    </td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ergebnis["ausnutzung"] <= 100.0:

        st.success(
            f"Nachweis erfüllt: Die Ausnutzung beträgt "
            f"{ergebnis['ausnutzung']:.1f} %."
        )

    st.subheader("Schnittgrößenverläufe")

    # Diagramm aus dem gespeicherten PNG anzeigen
    st.image(
        diagramm_png,
        use_container_width=True,
    )

    # PDF-Bericht erzeugen
    pdf_datei = erstelle_pdf(
        ergebnis,
        diagramm_png,
    )

    st.download_button(
        label="📄 PDF-Ergebnisbericht herunterladen",
        data=pdf_datei,
        file_name="Ergebnisbericht_Einfeldtraeger.pdf",
        mime="application/pdf",
        type="secondary",
        use_container_width=True,
    )

else:

    st.info(
        "Geben Sie die gewünschten Werte ein und klicken Sie anschließend "
        "auf „Berechnung durchführen“."
    )
