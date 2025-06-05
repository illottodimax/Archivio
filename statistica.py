import tkinter as tk
from tkinter import ttk, scrolledtext, font as tkfont, messagebox
import threading
import pandas as pd
from collections import defaultdict, Counter
from io import StringIO
import requests

# --- Configurazione GitHub (invariata) ---
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"
RUOTE_DISPONIBILI = {
    'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova',
    'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma',
    'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'
}
URL_RUOTE = {
    key: f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{value.upper()}.txt'
    for key, value in RUOTE_DISPONIBILI.items()
}

# --- FUNZIONI DI GENERAZIONE FORMAZIONI (invariate) ---
# (Omesse per brevità, sono identiche a prima)
def genera_formazioni_distanza_estratto():
    diz_formazioni_distanza = {}
    for d in range(1, 46):
        nome_key_formazione = f"Distanza_{d}"
        coppie_per_distanza_d = []
        for i in range(1, 90 - d + 1): n1, n2 = i, i + d; coppie_per_distanza_d.append((n1, n2))
        if coppie_per_distanza_d: diz_formazioni_distanza[nome_key_formazione] = coppie_per_distanza_d
    return diz_formazioni_distanza
def genera_diametrali_estratto(): return [(i, i + 45) for i in range(1, 46)]
def genera_diametrali_in_decina_estratto():
    f = []; [f.append(tuple(sorted((j, j + 5)))) for i in range(1, 90, 10) for j in range(i, i + 5) if j + 5 <= i + 9 and j + 5 <= 90]; return list(set(f))
def genera_coppie_somma_90_estratto(): return [(i, 90 - i) for i in range(1, 45)]
def genera_terzine_simmetriche_estratto(): return [(i, i + 30, i + 60) for i in range(1, 31)]
def get_vertibile_aggiornato(n):
    if not 1 <= n <= 90: return None
    if 11 <= n <= 88 and n % 11 == 0: return (n // 10) * 10 + 9
    if n == 90: return 9
    s = str(n).zfill(2); vert_s = s[::-1]; vert = int(vert_s)
    if vert_s.startswith('0') and len(vert_s) > 1 and vert != 0 : return vert
    elif vert == 0 and n % 10 == 0 and n != 0: return int(str(n)[0])
    if vert == 0 or vert > 90: return None
    return vert
def genera_coppie_vertibili_estratto_aggiornato():
    coppie_vertibili = set(); [(coppie_vertibili.add(tuple(sorted((i, v))))) for i in range(1, 91) if (v := get_vertibile_aggiornato(i)) is not None and 1 <= v <= 90 and i != v]; return list(coppie_vertibili)
def get_figura(n):
    if n == 90: return 9
    return (sum(int(d) for d in str(n)) - 1) % 9 + 1 if 1 <= n <= 89 else None
def genera_lista_gruppi_decine_naturali_ambo(): return [list(range(i * 10 + 1, (i + 1) * 10 + 1)) for i in range(9)]
def genera_lista_gruppi_decine_cabalistiche_ambo():
    lg = [list(range(1, 10)) + [90]]; lg.extend([list(range(i * 10, i * 10 + 10)) for i in range(1, 9)]); return lg
def genera_lista_gruppi_figure_ambo():
    gt = defaultdict(list); [gt[fig].append(i) for i in range(1, 91) if (fig := get_figura(i))]; return [sorted(gt[fn]) for fn in sorted(gt.keys())]
def genera_lista_gruppi_cadenze_ambo():
    gt = defaultdict(list); [gt[i % 10].append(i) for i in range(1, 91)]; return [sorted(gt[cn]) for cn in sorted(gt.keys())]
def genera_lista_gruppi_quindicine_ambo(): return [list(range(i * 15 + 1, (i + 1) * 15 + 1)) for i in range(6)]

# --- Funzioni di Utility (invariate) ---
# (Omesse per brevità)
def load_archivio_ruota(url):
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        estrazioni = []
        for line in StringIO(response.text).readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try: numeri_int = [int(n) for n in parts[-5:]]; estrazioni.append(numeri_int) if len(numeri_int) == 5 else None
                except ValueError: continue
        return estrazioni
    except requests.RequestException: return []
    except Exception: return []
def precalcola_presenze_singoli_numeri(estrazioni_ruota): c = Counter(); [c.update(e) for e in estrazioni_ruota]; return c

# --- Funzioni Core per Statistiche (invariate) ---
# (Omesse per brevità)
def calcola_statistiche_estratto(estrazioni_ruota, formazione_numeri, presenze_numeri_ruota):
    if not estrazioni_ruota: return {'Ritardo': 0, 'RitMax': 0, 'IncrRitMax': 0, 'Frequenza': 0, 'Presenze': 0, 'IndConv': float('inf'), 'Rit/RitMax': 0.0}
    fTEA, pFE, rC, idx = len(estrazioni_ruota), 0, 0, len(estrazioni_ruota) -1
    while idx >= 0:
        if any(n in estrazioni_ruota[idx] for n in formazione_numeri): break
        rC += 1; idx -=1
    if idx < 0: rC = fTEA
    lRC, rT = [], 0
    for d in estrazioni_ruota:
        if any(n in d for n in formazione_numeri): pFE += 1; lRC.append(rT) if rT > 0 else None; rT = 0
        else: rT += 1
    rMC = fTEA if pFE == 0 and rC == fTEA else max(max(lRC) if lRC else 0, rC)
    iRM = 0
    if rC == rMC:
        if pFE > 0:
            if lRC: sorted_lRC = sorted(lRC, reverse=True); iRM = (rC - sorted_lRC[1]) if len(sorted_lRC) > 1 and rC > sorted_lRC[1] else (rC if len(sorted_lRC) ==1 else 0)
            else: iRM = rC 
        else: iRM = rC 
    fU = sum(presenze_numeri_ruota.get(n, 0) for n in formazione_numeri); iC = float('inf')
    if pFE > 0 and fTEA > 0: mRT = fTEA / pFE; iC = (rC / mRT) if mRT > 0 else (float('inf') if rC > 0 else 0)
    elif pFE == 0 and rC > 0: iC = float('inf')
    rSRM = (rC / rMC) if rMC > 0 else 0.0
    return {'Ritardo': rC, 'RitMax': rMC, 'IncrRitMax': iRM, 'Frequenza': fU, 'Presenze': pFE, 'IndConv': iC, 'Rit/RitMax': rSRM}
def calcola_statistiche_singolo_numero_estratto(estrazioni_ruota, numero_da_analizzare, presenze_numeri_ruota):
    return calcola_statistiche_estratto(estrazioni_ruota, [numero_da_analizzare], presenze_numeri_ruota)
def calcola_statistiche_gruppo_ambo(estrazioni_ruota, gruppo_numeri_formazione, presenze_numeri_ruota):
    if not estrazioni_ruota or len(gruppo_numeri_formazione) < 2: return {'Ritardo': 0, 'RitMax': 0, 'IncrRitMax': 0, 'Frequenza': 0, 'Presenze': 0, 'IndConv': float('inf'), 'Rit/RitMax': 0.0}
    fTEA,pGA,rC,sG,idx = len(estrazioni_ruota),0,0,set(gruppo_numeri_formazione),len(estrazioni_ruota)-1
    while idx >= 0:
        if len(set(estrazioni_ruota[idx]).intersection(sG)) >= 2: break
        rC += 1; idx -= 1
    if idx < 0: rC = fTEA
    lRC, rT = [], 0
    for d in estrazioni_ruota:
        if len(set(d).intersection(sG)) >= 2: pGA += 1; lRC.append(rT) if rT > 0 else None; rT = 0
        else: rT += 1
    rMC = fTEA if pGA == 0 and rC == fTEA else max(max(lRC) if lRC else 0, rC)
    iRM = 0
    if rC == rMC:
        if pGA > 0:
            if lRC: sorted_lRC = sorted(lRC, reverse=True); iRM = (rC - sorted_lRC[1]) if len(sorted_lRC) > 1 and rC > sorted_lRC[1] else (rC if len(sorted_lRC) ==1 else 0)
            else: iRM = rC
        else: iRM = rC
    fU = sum(presenze_numeri_ruota.get(n, 0) for n in gruppo_numeri_formazione); iC = float('inf')
    if pGA > 0 and fTEA > 0: mRT = fTEA / pGA; iC = (rC / mRT) if mRT > 0 else (float('inf') if rC > 0 else 0)
    elif pGA == 0 and rC > 0: iC = float('inf')
    rSRM = (rC / rMC) if rMC > 0 else 0.0
    return {'Ritardo': rC, 'RitMax': rMC, 'IncrRitMax': iRM, 'Frequenza': fU, 'Presenze': pGA, 'IndConv': iC, 'Rit/RitMax': rSRM}
def calcola_statistiche_gruppo_terno(estrazioni_ruota, gruppo_numeri_formazione, presenze_numeri_ruota):
    if not estrazioni_ruota or len(gruppo_numeri_formazione) < 3: return {'Ritardo': 0, 'RitMax': 0, 'IncrRitMax': 0, 'Frequenza': 0, 'Presenze': 0, 'IndConv': float('inf'), 'Rit/RitMax': 0.0}
    fTEA,pGT,rC,sG,idx = len(estrazioni_ruota),0,0,set(gruppo_numeri_formazione),len(estrazioni_ruota)-1
    while idx >= 0:
        if len(set(estrazioni_ruota[idx]).intersection(sG)) >= 3: break
        rC += 1; idx -= 1
    if idx < 0: rC = fTEA
    lRC,rT = [],0
    for d in estrazioni_ruota:
        if len(set(d).intersection(sG)) >= 3: pGT += 1; lRC.append(rT) if rT > 0 else None; rT = 0
        else: rT += 1
    rMC = fTEA if pGT == 0 and rC == fTEA else max(max(lRC) if lRC else 0, rC)
    iRM = 0
    if rC == rMC:
        if pGT > 0:
            if lRC: sorted_lRC = sorted(lRC, reverse=True); iRM = (rC - sorted_lRC[1]) if len(sorted_lRC) > 1 and rC > sorted_lRC[1] else (rC if len(sorted_lRC) == 1 else 0)
            else: iRM = rC
        else: iRM = rC
    fU = sum(presenze_numeri_ruota.get(n,0) for n in gruppo_numeri_formazione); iC = float('inf')
    if pGT > 0 and fTEA > 0: mRT = fTEA / pGT; iC = (rC / mRT) if mRT > 0 else (float('inf') if rC > 0 else 0)
    elif pGT == 0 and rC > 0: iC = float('inf')
    rSRM = (rC / rMC) if rMC > 0 else 0.0
    return {'Ritardo': rC, 'RitMax': rMC, 'IncrRitMax': iRM, 'Frequenza': fU, 'Presenze': pGT, 'IndConv': iC, 'Rit/RitMax': rSRM}

# --- CLASSE SCROLLEDFRAME (invariata) ---
class ScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)
        self.vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, yscrollcommand=self.vscrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        self.vscrollbar.config(command=self.canvas.yview)
        self.interior = ttk.Frame(self.canvas)
        self.interior_id = self.canvas.create_window(0, 0, window=self.interior, anchor=tk.NW)
        self.interior.bind('<Configure>', self._on_interior_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
    def _on_interior_configure(self, event): self.canvas.config(scrollregion=self.canvas.bbox("all"))
    def _on_canvas_configure(self, event): self.canvas.itemconfig(self.interior_id, width=event.width)

class LottoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizzatore Statistiche Lotto Avanzato")
        self.root.geometry("1000x800") 

        self.style = ttk.Style()
        self.style.theme_use('clam') 
        
        self.title_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.style.configure("Title.TLabel", font=self.title_font, padding=(0, 5, 0, 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 10, "bold"))
        self.style.configure("Accent.TButton", foreground="white", background="#4CAF50", font=("Helvetica", 10, "bold"))
        self.style.map("Accent.TButton", background=[('active', '#45a049')])
        self.style.configure("Small.TButton", padding=(5,3), font=("Helvetica", 9))

        controls_outer_frame = ttk.Frame(root, padding="10")
        controls_outer_frame.pack(fill=tk.X)

        ruote_lf = ttk.Labelframe(controls_outer_frame, text="Selezione Ruote", padding="10")
        ruote_lf.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        self.ruote_vars = {}
        for i, (sigla, nome) in enumerate(RUOTE_DISPONIBILI.items()):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(ruote_lf, text=nome, variable=var)
            cb.grid(row=i % 6, column=i // 6, sticky=tk.W, padx=5, pady=2)
            self.ruote_vars[sigla] = var
        ruote_buttons_frame = ttk.Frame(ruote_lf)
        ruote_buttons_frame.grid(row=6, column=0, columnspan=2, pady=(10,0))
        ttk.Button(ruote_buttons_frame, text="Tutte", command=lambda: self.select_all_ruote(True), style="Small.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(ruote_buttons_frame, text="Nessuna", command=lambda: self.select_all_ruote(False), style="Small.TButton").pack(side=tk.LEFT, padx=2)

        action_status_frame = ttk.Frame(controls_outer_frame, padding="5")
        action_status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        analysis_buttons_lf = ttk.Labelframe(action_status_frame, text="Azioni di Analisi", padding="10")
        analysis_buttons_lf.pack(fill=tk.X, pady=(0,10))

        # --- MODIFICA TESTO PULSANTI DI COMANDO ---
        self.btn_estratto_principale = ttk.Button(analysis_buttons_lf, text="Analisi per Estratto", command=lambda: self._start_analysis_runner(self._run_analisi_estratto_principale), style="Accent.TButton")
        self.btn_estratto_principale.pack(fill=tk.X, pady=2)
        
        self.btn_coppie_distanza_cmd = ttk.Button(analysis_buttons_lf, text="Analisi per Estratto su Coppie a Distanza", command=lambda: self._start_analysis_runner(self._run_analisi_coppie_distanza), style="Accent.TButton")
        self.btn_coppie_distanza_cmd.pack(fill=tk.X, pady=2)

        self.btn_gruppi_ambo = ttk.Button(analysis_buttons_lf, text="Analisi Gruppi per Ambo", command=lambda: self._start_analysis_runner(self._run_analisi_gruppi_ambo), style="Accent.TButton")
        self.btn_gruppi_ambo.pack(fill=tk.X, pady=2)
        
        self.btn_gruppi_terno = ttk.Button(analysis_buttons_lf, text="Analisi Gruppi per Terno", command=lambda: self._start_analysis_runner(self._run_analisi_gruppi_terno), style="Accent.TButton")
        self.btn_gruppi_terno.pack(fill=tk.X, pady=2)
        
        self.analysis_buttons = [self.btn_estratto_principale, self.btn_coppie_distanza_cmd, self.btn_gruppi_ambo, self.btn_gruppi_terno]
        # --- FINE MODIFICA TESTO PULSANTI DI COMANDO ---


        self.status_label = ttk.Label(action_status_frame, text="Pronto.", anchor="w", font=("Helvetica", 9))
        self.status_label.pack(fill=tk.X, pady=(5,5))
        self.progress_bar = ttk.Progressbar(action_status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0,5))
        
        self.log_text_area = scrolledtext.ScrolledText(action_status_frame, wrap=tk.WORD, height=3, font=("Courier New", 8))
        self.log_text_area.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        self.log_text_area.insert(tk.END, "Log caricamento archivi:\n")
        self.log_text_area.config(state=tk.DISABLED)

        self.notebook = ttk.Notebook(root, padding="5")
        self.notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.tab_estratto_principale = ScrolledFrame(self.notebook)
        self.tab_coppie_distanza = ScrolledFrame(self.notebook)
        self.tab_ambo = ScrolledFrame(self.notebook)
        self.tab_terno = ScrolledFrame(self.notebook)
        
        # --- MODIFICA TESTO SCHEDE NOTEBOOK ---
        self.notebook.add(self.tab_estratto_principale, text='Analisi per Estratto')
        self.notebook.add(self.tab_coppie_distanza, text='Estratto su Coppie a Distanza')
        self.notebook.add(self.tab_ambo, text='Gruppi per Ambo')
        self.notebook.add(self.tab_terno, text='Gruppi per Terno')
        # --- FINE MODIFICA TESTO SCHEDE NOTEBOOK ---
        
        self.treeviews = {}
        self.cols_rep = ['Nome', 'Ruota', 'Numeri', 'Ritardo', 'RitMax', 'IncrRitMax', 'Frequenza', 'Presenze', 'IndConv', 'Rit/RitMax']
        self.cols_align_center = ['Ritardo', 'RitMax', 'IncrRitMax', 'Frequenza', 'Presenze']
        self.cols_align_right = ['IndConv', 'Rit/RitMax']
        
        self.archivi_ruote = {}
        self.presenze_singoli_per_ruota = {}
        self.archivi_caricati_per_sigle = set()

        self.pregenerate_formations()

    def pregenerate_formations(self): # Logica interna invariata
        self.KEY_ESTRATTI_SEMPLICI = "EstrattiSemplici"
        self.formazioni_estratto_principale_keys = [
            self.KEY_ESTRATTI_SEMPLICI, 
            "VertibiliE", "DiametraliE", "DiametraliDecinaE", "CoppieSomma90E", "TerzineSimmetricheE"
        ]
        self.formazioni_estratto_distanza_keys = [f"Distanza_{d}" for d in range(1, 46)]
        all_estratto_defs = {
            "DiametraliE": genera_diametrali_estratto(), "DiametraliDecinaE": genera_diametrali_in_decina_estratto(),
            "CoppieSomma90E": genera_coppie_somma_90_estratto(), "TerzineSimmetricheE": genera_terzine_simmetriche_estratto(),
            "VertibiliE": genera_coppie_vertibili_estratto_aggiornato()}
        formazioni_per_distanza = genera_formazioni_distanza_estratto()
        all_estratto_defs.update(formazioni_per_distanza)
        self.formazioni_estratto_defs = all_estratto_defs
        self.nomi_display_estratto = { # I nomi per i titoli dei Treeview rimangono più descrittivi
            self.KEY_ESTRATTI_SEMPLICI: "Estratti Semplici (Top 10 Rit. Globali)", 
            "VertibiliE": "Coppie Vertibili",
            "DiametraliE": "Diametrali", 
            "DiametraliDecinaE": "Diam. in Decina", 
            "CoppieSomma90E": "Coppie Somma 90",
            "TerzineSimmetricheE": "Terzine Simmetriche"}
        for d in range(1, 46): self.nomi_display_estratto[f"Distanza_{d}"] = f"Coppie Distanza {d}"
        self.categorie_gruppi_amboterno_defs = {
            "DecineNaturali": genera_lista_gruppi_decine_naturali_ambo(), 
            "DecineCabalistiche": genera_lista_gruppi_decine_cabalistiche_ambo(),
            "Figure": genera_lista_gruppi_figure_ambo(), 
            "Cadenze": genera_lista_gruppi_cadenze_ambo(), 
            "Quindicine": genera_lista_gruppi_quindicine_ambo()
        }
        self.nomi_display_gruppi_ambo = {
            "DecineNaturali": "Decine Naturali (Ambo)", "DecineCabalistiche": "Decine Cabalistiche (Ambo)", 
            "Figure": "Figure (Ambo)", "Cadenze": "Cadenze (Ambo)", "Quindicine": "Quindicine (Ambo)"
        }
        self.nomi_display_gruppi_terno = {
            "DecineNaturali": "Decine Naturali (Terno)", "DecineCabalistiche": "Decine Cabalistiche (Terno)", 
            "Figure": "Figure (Terno)", "Cadenze": "Cadenze (Terno)", "Quindicine": "Quindicine (Terno)"
        }

    # ... (tutti gli altri metodi: select_all_ruote, _append_log, update_status, create_treeview_in_tab, fmt_flt_for_gui, populate_treeview, _set_buttons_state, _start_analysis_runner, _ensure_archivi_caricati_sync, _clear_treeviews_for_keys, _run_analisi_estratto_principale, _run_analisi_coppie_distanza, _run_analisi_gruppi_ambo, _run_analisi_gruppi_terno sono INVARIATI rispetto all'ultima versione completa e OMESSI PER BREVITÀ qui, ma devono essere presenti nel file finale)
    def select_all_ruote(self, select_state):
        for var in self.ruote_vars.values(): var.set(select_state)
    def _append_log(self, message):
        self.log_text_area.config(state=tk.NORMAL)
        self.log_text_area.insert(tk.END, message + "\n")
        self.log_text_area.see(tk.END)
        self.log_text_area.config(state=tk.DISABLED)
        self.root.update_idletasks()
    def update_status(self, message, progress_value=None):
        self.status_label.config(text=message)
        if progress_value is not None: self.progress_bar['value'] = progress_value
        self.root.update_idletasks()
    def create_treeview_in_tab(self, scrolled_tab_frame, category_title):
        outer_frame = ttk.Frame(scrolled_tab_frame.interior) 
        outer_frame.pack(pady=(5,10), padx=5, fill=tk.X, expand=False) 
        title_label = ttk.Label(outer_frame, text=category_title, style="Header.TLabel")
        title_label.pack(anchor=tk.W, pady=(0, 5))
        tv_frame = ttk.Frame(outer_frame)
        tv_frame.pack(fill=tk.X, expand=True)
        tree = ttk.Treeview(tv_frame, columns=self.cols_rep, show='headings', height=5) 
        for col in self.cols_rep:
            tree.heading(col, text=col)
            width = tkfont.Font().measure(col) + 20 
            if col == 'Numeri': width = max(width, 150)
            elif col == 'Nome': width = max(width, 150)
            elif col == 'Ruota': width = max(width, 80)
            else: width = max(width, 70) 
            alignment = 'w';
            if col in self.cols_align_center: alignment = 'center'
            elif col in self.cols_align_right: alignment = 'e'
            tree.column(col, width=width, anchor=alignment, stretch=tk.NO)
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y); hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True); return tree
    def fmt_flt_for_gui(self, x):
        if pd.isna(x): return 'NaN'
        if x == float('inf'): return 'inf'
        s = f"{x:.4f}" 
        if '.' in s: i, d = s.split('.'); return i if d == '0000' else (f"{i}.{d[0]}" if d.endswith('000') else (f"{i}.{d[:2]}" if d.endswith('00') else (f"{i}.{d[:3]}" if d.endswith('0') else f"{i}.{d}")))
        return f"{int(x)}"

    def populate_treeview(self, tree, df, cols_to_format_float):
        if tree is None: return 
        
        # Configura i tag
        tree.tag_configure('ritardo_max_raggiunto', foreground='#D32F2F', font=tkfont.Font(weight="bold")) # Rosso scuro, grassetto
        # --- MODIFICA COLORE E FONT PER IL VERDE ---
        tree.tag_configure('vicino_ritardo_max', foreground='#4CAF50', font=tkfont.Font(weight="bold")) # Verde brillante (come i pulsanti), grassetto
        # --- FINE MODIFICA ---

        for item in tree.get_children(): tree.delete(item)
        
        df_display = df.copy()
        
        numeric_cols_for_tags = ['Ritardo', 'RitMax']
        for col in numeric_cols_for_tags:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0)

        for col_format in cols_to_format_float:
            if col_format in df_display.columns:
                 df_display[col_format] = df[col_format].apply(self.fmt_flt_for_gui)
        
        for index, row in df.iterrows(): 
            tag_to_apply = () 

            ritardo_val = row['Ritardo']
            ritmax_val = row['RitMax']

            is_numeric_ritardo = isinstance(ritardo_val, (int, float))
            is_numeric_ritmax = isinstance(ritmax_val, (int, float))

            if is_numeric_ritardo and is_numeric_ritmax:
                if ritmax_val > 0 and ritardo_val == ritmax_val:
                    tag_to_apply = ('ritardo_max_raggiunto',)
                elif ritmax_val > 0 and ritardo_val >= 0.85 * ritmax_val and ritardo_val < ritmax_val: 
                    tag_to_apply = ('vicino_ritardo_max',)
            
            display_values = []
            for col_name in self.cols_rep:
                if col_name in df_display.columns:
                    # Usa il valore da df_display se la colonna è stata formattata
                    # Questo è importante per le colonne float
                    if col_name in cols_to_format_float:
                         display_values.append(df_display.loc[index, col_name])
                    else: # Per colonne non float (come Nome, Ruota, Numeri interi non formattati) usa il valore originale
                         display_values.append(row[col_name])
                elif col_name in row: 
                    display_values.append(row[col_name])
                else:
                    display_values.append("") 

            tree.insert("", tk.END, values=display_values, tags=tag_to_apply)

    def _set_buttons_state(self, state):
        for btn in self.analysis_buttons: btn.config(state=state)
    def _start_analysis_runner(self, analysis_function_to_run):
        self._set_buttons_state(tk.DISABLED)
        self.progress_bar['value'] = 0
        self.update_status("Avvio analisi...")
        def threaded_task():
            try:
                if not self._ensure_archivi_caricati_sync(): return
                analysis_function_to_run()
                self.update_status("Analisi specifica completata.", 100)
                messagebox.showinfo("Completato", "L'analisi selezionata è terminata.")
            except Exception as e:
                self.update_status(f"Errore: {e}", self.progress_bar['value'])
                messagebox.showerror("Errore", f"Si è verificato un errore: {e}")
                import traceback
                self._append_log(f"ERRORE CRITICO: {e}\n{traceback.format_exc()}")
            finally:
                self._set_buttons_state(tk.NORMAL)
                self.progress_bar['value'] = 0
        analysis_thread = threading.Thread(target=threaded_task); analysis_thread.daemon = True; analysis_thread.start()
    def _ensure_archivi_caricati_sync(self):
        current_selected_sigle = {sigla for sigla, var in self.ruote_vars.items() if var.get()}
        if not current_selected_sigle:
            self.root.after(0, lambda: messagebox.showwarning("Selezione mancante", "Selezionare almeno una ruota."))
            return False
        if self.archivi_caricati_per_sigle == current_selected_sigle and self.archivi_ruote:
            self.root.after(0, lambda: self._append_log("Archivi già caricati per le ruote selezionate."))
            return True
        self.root.after(0, lambda: self.update_status("Caricamento archivi...", 5))
        self.archivi_ruote.clear(); self.presenze_singoli_per_ruota.clear(); self.archivi_caricati_per_sigle.clear()
        ruote_da_caricare = {sigla: RUOTE_DISPONIBILI[sigla] for sigla in current_selected_sigle}
        total_ruote_to_load = len(ruote_da_caricare); loaded_count = 0
        progress_per_ruota_load = 50 / total_ruote_to_load if total_ruote_to_load > 0 else 0
        for sigla_r, nome_r_comp in ruote_da_caricare.items():
            self.root.after(0, lambda s=sigla_r, n=nome_r_comp, lc=loaded_count, pprl=progress_per_ruota_load: self.update_status(f"Caricamento: {n}...", 5 + (lc * pprl)))
            self.root.after(0, lambda n=nome_r_comp: self._append_log(f" Tentativo caricamento: {n}..."))
            est = load_archivio_ruota(URL_RUOTE[sigla_r])
            if est:
                self.archivi_ruote[nome_r_comp] = est
                self.presenze_singoli_per_ruota[nome_r_comp] = precalcola_presenze_singoli_numeri(est)
                self.archivi_caricati_per_sigle.add(sigla_r)
                self.root.after(0, lambda n=nome_r_comp, l=len(est): self._append_log(f"  OK: {n} ({l} estrazioni)."))
            else: self.root.after(0, lambda n=nome_r_comp: self._append_log(f"  FALLITO: Nessun dato per {n}."))
            loaded_count += 1
        if not self.archivi_ruote:
            self.root.after(0, lambda: self.update_status("Errore: Nessun archivio caricato.", 0))
            self.root.after(0, lambda: messagebox.showerror("Errore", "Nessun archivio caricato."))
            return False
        self.root.after(0, lambda: self._append_log("Caricamento archivi completato."))
        return True
    def _clear_treeviews_for_keys(self, keys_to_clear, target_tab_interior=None):
        for key in keys_to_clear:
            if key in self.treeviews:
                if target_tab_interior is None or self.treeviews[key].master.master.master == target_tab_interior:
                    for item in self.treeviews[key].get_children(): 
                        self.treeviews[key].delete(item)
    def _run_analisi_estratto_principale(self):
        self.notebook.select(self.tab_estratto_principale)
        self._clear_treeviews_for_keys(self.formazioni_estratto_principale_keys, self.tab_estratto_principale.interior) 
        self.update_status("Analisi per Estratto in corso...", 55)
        num_ruote = len(self.archivi_ruote)
        total_calcs_ops = num_ruote * 90 
        for k_tipo_loop in self.formazioni_estratto_principale_keys:
            if k_tipo_loop != self.KEY_ESTRATTI_SEMPLICI and k_tipo_loop in self.formazioni_estratto_defs:
                total_calcs_ops += len(self.formazioni_estratto_defs[k_tipo_loop]) * num_ruote
        calcs_done_ops = 0
        progress_per_op = 45 / total_calcs_ops if total_calcs_ops > 0 else 0
        for k_tipo in self.formazioni_estratto_principale_keys:
            nome_d = self.nomi_display_estratto.get(k_tipo, k_tipo)
            self.root.after(0, lambda nd=nome_d: self.update_status(f"Estratto: Elaboro {nd}..."))
            created_event = threading.Event()
            def create_tv_gui():
                if k_tipo not in self.treeviews or self.treeviews[k_tipo].master.master.master != self.tab_estratto_principale.interior:
                    setattr(self, 'treeviews', {**self.treeviews, k_tipo: self.create_treeview_in_tab(self.tab_estratto_principale, nome_d)})
                created_event.set()
            if k_tipo not in self.treeviews or self.treeviews[k_tipo].master.master.master != self.tab_estratto_principale.interior: self.root.after(0, create_tv_gui); created_event.wait()
            current_results_for_category = []
            if k_tipo == self.KEY_ESTRATTI_SEMPLICI:
                all_stats_globali = []
                for nome_r, estr_r in self.archivi_ruote.items():
                    pres_num_r = self.presenze_singoli_per_ruota[nome_r]
                    for numero in range(1, 91):
                        stats = calcola_statistiche_singolo_numero_estratto(estr_r, numero, pres_num_r)
                        all_stats_globali.append({'Nome': nome_d, 'Ruota': nome_r, 'Numeri': f"{numero:02d}", **stats})
                        calcs_done_ops += 1
                        current_progress = 55 + (calcs_done_ops * progress_per_op)
                        self.root.after(0, lambda p=current_progress: self.progress_bar.config(value=p))
                df_globali_ordinati = pd.DataFrame(all_stats_globali).sort_values(by=['Ritardo','RitMax'], ascending=[False,False])
                current_results_for_category = df_globali_ordinati.head(10).to_dict('records')
            else: 
                if k_tipo not in self.formazioni_estratto_defs: continue
                lista_f = self.formazioni_estratto_defs[k_tipo]
                for nome_r, estr_r in self.archivi_ruote.items():
                    pres_num_r = self.presenze_singoli_per_ruota[nome_r]
                    for form_n in lista_f:
                        stats = calcola_statistiche_estratto(estr_r, form_n, pres_num_r)
                        current_results_for_category.append({'Nome': nome_d, 'Ruota': nome_r, 'Numeri': ' '.join(f"{n:02d}" for n in sorted(list(form_n))), **stats})
                        calcs_done_ops +=1
                        current_progress = 55 + (calcs_done_ops * progress_per_op)
                        self.root.after(0, lambda p=current_progress: self.progress_bar.config(value=p))
            if current_results_for_category:
                if k_tipo == self.KEY_ESTRATTI_SEMPLICI: df_display = pd.DataFrame(current_results_for_category) 
                else: df_display = pd.DataFrame(current_results_for_category).sort_values(by=['Ritardo','RitMax'], ascending=[False,False]).head(10)
                self.root.after(0, lambda tv=self.treeviews.get(k_tipo), d=df_display: self.populate_treeview(tv, d, ['IndConv','Rit/RitMax']))
            current_results_for_category.clear()
    def _run_analisi_coppie_distanza(self):
        self.notebook.select(self.tab_coppie_distanza)
        self._clear_treeviews_for_keys(self.formazioni_estratto_distanza_keys, self.tab_coppie_distanza.interior)
        self.update_status("Analisi per Estratto su Coppie a Distanza in corso...", 55)
        num_ruote = len(self.archivi_ruote); total_calcs_ops = 0
        for k_tipo_loop in self.formazioni_estratto_distanza_keys:
            if k_tipo_loop in self.formazioni_estratto_defs: total_calcs_ops += len(self.formazioni_estratto_defs[k_tipo_loop]) * num_ruote
        calcs_done_ops = 0; progress_per_op = 45 / total_calcs_ops if total_calcs_ops > 0 else 0
        for k_tipo in self.formazioni_estratto_distanza_keys:
            if k_tipo not in self.formazioni_estratto_defs: continue
            lista_f = self.formazioni_estratto_defs[k_tipo]; nome_d = self.nomi_display_estratto.get(k_tipo, k_tipo)
            self.root.after(0, lambda nd=nome_d: self.update_status(f"Distanze: Elaboro {nd}..."))
            created_event = threading.Event()
            def create_tv_gui(): 
                if k_tipo not in self.treeviews or self.treeviews[k_tipo].master.master.master != self.tab_coppie_distanza.interior:
                    setattr(self, 'treeviews', {**self.treeviews, k_tipo: self.create_treeview_in_tab(self.tab_coppie_distanza, nome_d)})
                created_event.set()
            if k_tipo not in self.treeviews or self.treeviews[k_tipo].master.master.master != self.tab_coppie_distanza.interior: self.root.after(0, create_tv_gui); created_event.wait()
            current_results_for_category = []
            for nome_r, estr_r in self.archivi_ruote.items():
                pres_num_r = self.presenze_singoli_per_ruota[nome_r]
                for form_n in lista_f:
                    stats = calcola_statistiche_estratto(estr_r, form_n, pres_num_r)
                    current_results_for_category.append({'Nome': nome_d, 'Ruota': nome_r, 'Numeri': ' '.join(f"{n:02d}" for n in sorted(list(form_n))), **stats})
                    calcs_done_ops +=1
                    current_progress = 55 + (calcs_done_ops * progress_per_op)
                    self.root.after(0, lambda p=current_progress: self.progress_bar.config(value=p))
            if current_results_for_category:
                df = pd.DataFrame(current_results_for_category).sort_values(by=['Ritardo','RitMax'], ascending=[False,False]).head(10)
                self.root.after(0, lambda tv=self.treeviews.get(k_tipo), d=df: self.populate_treeview(tv, d, ['IndConv','Rit/RitMax']))
            current_results_for_category.clear()
    def _run_analisi_gruppi_ambo(self):
        self.notebook.select(self.tab_ambo)
        self._clear_treeviews_for_keys(self.categorie_gruppi_amboterno_defs.keys(), self.tab_ambo.interior)
        self.update_status("Analisi Gruppi per Ambo in corso...", 55)
        num_ruote = len(self.archivi_ruote); total_calcs_ops = sum(len(g) for g in self.categorie_gruppi_amboterno_defs.values()) * num_ruote
        calcs_done_ops = 0; progress_per_op = 45 / total_calcs_ops if total_calcs_ops > 0 else 0
        for categoria_gruppo, lista_di_gruppi_specifici in self.categorie_gruppi_amboterno_defs.items():
            nome_display_categoria = self.nomi_display_gruppi_ambo.get(categoria_gruppo, categoria_gruppo)
            self.root.after(0, lambda ndc=nome_display_categoria: self.update_status(f"Ambo: Elaboro {ndc}..."))
            created_event = threading.Event()
            def create_tv_gui(): 
                if categoria_gruppo not in self.treeviews or self.treeviews[categoria_gruppo].master.master.master != self.tab_ambo.interior:
                    setattr(self, 'treeviews', {**self.treeviews, categoria_gruppo: self.create_treeview_in_tab(self.tab_ambo, nome_display_categoria)})
                created_event.set()
            if categoria_gruppo not in self.treeviews or self.treeviews[categoria_gruppo].master.master.master != self.tab_ambo.interior : self.root.after(0, create_tv_gui); created_event.wait()
            current_results_for_category = []
            for numeri_del_gruppo_specifico in lista_di_gruppi_specifici:
                for nome_r, estr_r in self.archivi_ruote.items():
                    pres_num_r = self.presenze_singoli_per_ruota[nome_r]
                    stats = calcola_statistiche_gruppo_ambo(estr_r, numeri_del_gruppo_specifico, pres_num_r)
                    current_results_for_category.append({'Nome': nome_display_categoria, 'Ruota': nome_r, 'Numeri': ' '.join(f"{n:02d}" for n in numeri_del_gruppo_specifico), **stats})
                calcs_done_ops +=1 
                current_progress = 55 + (calcs_done_ops * progress_per_op)
                self.root.after(0, lambda p=current_progress: self.progress_bar.config(value=p))
            if current_results_for_category:
                df = pd.DataFrame(current_results_for_category).sort_values(by=['Ritardo','RitMax'], ascending=[False,False]).head(10)
                self.root.after(0, lambda tv=self.treeviews.get(categoria_gruppo), d=df: self.populate_treeview(tv, d, ['IndConv','Rit/RitMax']))
            current_results_for_category.clear()
    def _run_analisi_gruppi_terno(self):
        self.notebook.select(self.tab_terno)
        keys_terno = [f"{k}_Terno" for k in self.categorie_gruppi_amboterno_defs.keys()]
        self._clear_treeviews_for_keys(keys_terno, self.tab_terno.interior)
        self.update_status("Analisi Gruppi per Terno in corso...", 55)
        num_ruote = len(self.archivi_ruote); total_calcs_ops = sum(len(g) for g in self.categorie_gruppi_amboterno_defs.values()) * num_ruote
        calcs_done_ops = 0; progress_per_op = 45 / total_calcs_ops if total_calcs_ops > 0 else 0
        for categoria_gruppo_base, lista_di_gruppi_specifici in self.categorie_gruppi_amboterno_defs.items():
            nome_display_categoria = self.nomi_display_gruppi_terno.get(categoria_gruppo_base, categoria_gruppo_base)
            chiave_treeview_terno = f"{categoria_gruppo_base}_Terno"
            self.root.after(0, lambda ndc=nome_display_categoria: self.update_status(f"Terno: Elaboro {ndc}..."))
            created_event = threading.Event()
            def create_tv_gui(): 
                if chiave_treeview_terno not in self.treeviews or self.treeviews[chiave_treeview_terno].master.master.master != self.tab_terno.interior:
                    setattr(self, 'treeviews', {**self.treeviews, chiave_treeview_terno: self.create_treeview_in_tab(self.tab_terno, nome_display_categoria)})
                created_event.set()
            if chiave_treeview_terno not in self.treeviews or self.treeviews[chiave_treeview_terno].master.master.master != self.tab_terno.interior: self.root.after(0, create_tv_gui); created_event.wait()
            current_results_for_category = []
            for numeri_del_gruppo_specifico in lista_di_gruppi_specifici:
                if len(numeri_del_gruppo_specifico) < 3: continue
                for nome_r, estr_r in self.archivi_ruote.items():
                    pres_num_r = self.presenze_singoli_per_ruota[nome_r]
                    stats = calcola_statistiche_gruppo_terno(estr_r, numeri_del_gruppo_specifico, pres_num_r)
                    current_results_for_category.append({'Nome': nome_display_categoria, 'Ruota': nome_r, 'Numeri': ' '.join(f"{n:02d}" for n in numeri_del_gruppo_specifico), **stats})
                calcs_done_ops +=1
                current_progress = 55 + (calcs_done_ops * progress_per_op)
                self.root.after(0, lambda p=current_progress: self.progress_bar.config(value=p))
            if current_results_for_category:
                df = pd.DataFrame(current_results_for_category).sort_values(by=['Ritardo','RitMax'], ascending=[False,False]).head(10)
                self.root.after(0, lambda tv=self.treeviews.get(chiave_treeview_terno), d=df: self.populate_treeview(tv, d, ['IndConv','Rit/RitMax']))
            current_results_for_category.clear()


if __name__ == '__main__':
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()