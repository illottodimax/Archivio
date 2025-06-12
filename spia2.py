import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkinter import scrolledtext
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import traceback
import sys
import itertools
from collections import Counter, OrderedDict
import urllib.request
import urllib.error

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# --- Configurazione GitHub ---
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"
RUOTE_NOMI_MAPPATURA = {
    'BARI': 'Bari', 'CAGLIARI': 'Cagliari', 'FIRENZE': 'Firenze', 'GENOVA': 'Genova',
    'MILANO': 'Milano', 'NAPOLI': 'Napoli', 'PALERMO': 'Palermo', 'ROMA': 'Roma',
    'TORINO': 'Torino', 'VENEZIA': 'Venezia', 'NAZIONALE': 'Nazionale'
}
URL_RUOTE = {
    key.upper(): f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{value.upper()}.txt'
    for key, value in RUOTE_NOMI_MAPPATURA.items()
}
# --- Fine Configurazione GitHub ---

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}
MAX_COLPI_GIOCO = 90

# Variabili GUI globali
data_source_var = None
cartella_frame = None
cartella_entry = None
btn_sfoglia = None
button_visualizza = None
button_verifica_futura = None
estrazioni_entry_verifica_futura = None
button_verifica_mista = None
text_combinazioni_miste = None
estrazioni_entry_verifica_mista = None

# Variabili per analisi posizionale e GUI
tipo_spia_var_global = None
entry_numeri_spia = []
combo_posizione_spia = None

# Variabili GUI per Numeri Simpatici
entry_numero_target_simpatici = None
listbox_ruote_simpatici = None
entry_top_n_simpatici = None

# Variabili GUI per il numero di risultati da mostrare nel popup
entry_num_estratti_popup = None
entry_num_ambi_popup = None
entry_num_terni_popup = None
entry_num_terni_per_ambo_popup = None
entry_num_quartine_per_ambo_popup = None
entry_num_cinquine_per_ambo_popup = None

# Variabili per le checkbox di ottimizzazione
calcola_ritardo_globale_var = None


# =============================================================================
# FUNZIONI GRAFICHE (Invariate)
# =============================================================================
def crea_grafico_barre(risultato, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list):
            numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple):
            numeri_spia_str = "-".join(map(str,spia_val))
        else:
            numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        res_tipo_analisi = risultato.get(tipo_analisi, {})
        if not res_tipo_analisi:
             ax.text(0.5, 0.5, f"Nessun dato per {tipo_analisi.upper()}", ha='center', va='center')
             ax.set_title(f"{tipo_stat.capitalize()} {tipo_analisi.upper()} su {ruota_verifica}")
             ax.axis('off'); plt.close(fig); return None
        if tipo_stat == "presenza":
            dati_serie = res_tipo_analisi.get('presenza', {}).get('top', pd.Series(dtype='float64'))
            percentuali_serie = res_tipo_analisi.get('presenza', {}).get('percentuali', pd.Series(dtype='float64'))
            base_conteggio = risultato.get('totale_eventi_spia', 0)
            titolo = f"Presenza {tipo_analisi.upper()} su {ruota_verifica}\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Serie Eventi Spia ({base_conteggio} totali)"
        else:
            dati_serie = res_tipo_analisi.get('frequenza', {}).get('top', pd.Series(dtype='float64'))
            percentuali_serie = res_tipo_analisi.get('frequenza', {}).get('percentuali', pd.Series(dtype='float64'))
            base_conteggio = sum(dati_serie.values) if not dati_serie.empty else 0
            titolo = f"Frequenza {tipo_analisi.upper()} su {ruota_verifica}\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Occorrenze Totali ({tipo_analisi.capitalize()})"
        dati = dati_serie.to_dict(); percentuali = percentuali_serie.to_dict()
        if tipo_analisi in ["ambo", "terno"]:
            numeri = list(dati.keys())
            valori = list(dati.values())
            perc = [percentuali.get(k, 0.0) for k in numeri]
        else:
            numeri = list(dati.keys()); valori = list(dati.values())
            perc = [percentuali.get(num, 0.0) for num in numeri]
        if not numeri:
            ax.text(0.5, 0.5, f"Nessun dato per {tipo_analisi.upper()}", ha='center', va='center')
            ax.set_title(titolo); ax.axis('off'); plt.close(fig); return None
        bars = ax.bar(numeri, valori, color='skyblue', width=0.6)
        for i, (bar, p_val) in enumerate(zip(bars, perc)):
            h = bar.get_height()
            p_txt = f'{p_val:.1f}%' if p_val > 0.01 else ''
            ax.text(bar.get_x() + bar.get_width()/2., h + max(valori or [1])*0.01, p_txt, ha='center', va='bottom', fontweight='bold', fontsize=8)
        ax.set_xlabel(f'{tipo_analisi.capitalize()} su ' + ruota_verifica); ax.set_ylabel(ylabel)
        ax.set_title(titolo, fontsize=11)
        ax.set_ylim(0, max(valori or [1]) * 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.tick_params(axis='x', rotation=60, labelsize=8)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Eventi Spia: {risultato.get('totale_eventi_spia', 0)}"
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=8)
        plt.tight_layout(pad=3.5); return fig
    except Exception as e:
        print(f"Errore in crea_grafico_barre ({tipo_analisi}, {tipo_stat}): {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def crea_tabella_lotto(risultato, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    if tipo_analisi != "estratto": return None
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list): numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple): numeri_spia_str = "-".join(map(str,spia_val))
        else: numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        n_eventi_spia = risultato.get('totale_eventi_spia', 0)
        numeri_lotto = np.arange(1, 91).reshape(9, 10)
        res_estratti = risultato.get('estratto', {})
        if tipo_stat == "presenza":
            percentuali_serie = res_estratti.get('all_percentuali_presenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Presenza ESTRATTI su {ruota_verifica}\n(dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        else:
            percentuali_serie = res_estratti.get('all_percentuali_frequenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Frequenza ESTRATTI su {ruota_verifica}\n(dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        percentuali = percentuali_serie.to_dict() if not percentuali_serie.empty else {}
        colors_norm = np.full(numeri_lotto.shape, 0.9)
        valid_perc = [p for p in percentuali.values() if pd.notna(p) and p > 0]
        max_perc = max(valid_perc) if valid_perc else 1.0
        if max_perc == 0: max_perc = 1.0
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]; num_str = str(num).zfill(2); num_str_alt = str(num)
                perc_val = percentuali.get(num_str)
                if perc_val is None: perc_val = percentuali.get(num_str_alt)
                if perc_val is not None and pd.notna(perc_val) and perc_val > 0: colors_norm[i, j] = 0.9 - (0.9 * (perc_val / max_perc))
        for r_idx in range(10):
            ax.axvline(r_idx - 0.5, color='gray', linestyle='-', alpha=0.3); ax.axhline(r_idx - 0.5, color='gray', linestyle='-', alpha=0.3)
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]; norm_color = colors_norm[i, j]
                cell_color = "white"; text_color = "black"
                if norm_color < 0.9:
                    intensity = (0.9 - norm_color) / 0.9
                    r_val = int(220 * (1 - intensity)); g_val = int(230 * (1 - intensity)); b_val = int(255 * (1 - intensity / 2))
                    cell_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    if intensity > 0.6: text_color = "white"
                edge_color = 'black' if norm_color < 0.9 else 'gray'
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cell_color, alpha=1.0, edgecolor=edge_color, linewidth=1)
                ax.add_patch(rect); ax.text(j, i, num, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.5, 9.5); ax.set_ylim(8.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.title(titolo, fontsize=14, pad=15)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Eventi Spia: {n_eventi_spia}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]); return fig
    except Exception as e:
        print(f"Errore in crea_tabella_lotto: {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def crea_heatmap_correlazione(risultati, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    if tipo_analisi != "estratto": return None
    fig = None
    try:
        if len(risultati) < 2: return None
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list): numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple): numeri_spia_str = "-".join(map(str,spia_val))
        else: numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        all_numeri_estratti = set(); percentuali_per_ruota = {}
        for ruota_v, _, res in risultati:
            if res is None or not isinstance(res, dict): continue
            res_tipo = res.get('estratto', {});
            perc_dict_serie = res_tipo.get(f'all_percentuali_{tipo_stat}', pd.Series(dtype='float64'))
            if perc_dict_serie.empty: continue
            perc_dict = perc_dict_serie.to_dict()
            if not perc_dict: continue
            percentuali_per_ruota[ruota_v] = {num: perc for num, perc in perc_dict.items() if pd.notna(perc) and perc > 0}
            if percentuali_per_ruota[ruota_v]: all_numeri_estratti.update(percentuali_per_ruota[ruota_v].keys())
        if not percentuali_per_ruota or not all_numeri_estratti: return None
        all_numeri_sorted = sorted(list(all_numeri_estratti), key=str); ruote_heatmap = sorted(list(percentuali_per_ruota.keys()))
        if len(ruote_heatmap) < 2: return None
        matrice = np.array([[percentuali_per_ruota[r].get(n, np.nan) for n in all_numeri_sorted] for r in ruote_heatmap])
        fig, ax = plt.subplots(figsize=(min(18, len(all_numeri_sorted)*0.5+2), max(4, len(ruote_heatmap)*0.4+1)))
        max_val = np.nanmax(matrice) if not np.all(np.isnan(matrice)) else 1.0; max_val = max(max_val, 1.0)
        if SEABORN_AVAILABLE:
            sns.heatmap(matrice, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=all_numeri_sorted, yticklabels=ruote_heatmap, ax=ax, linewidths=.5, linecolor='gray', cbar=True, vmin=0, vmax=max_val, annot_kws={"size":7})
            ax.tick_params(axis='x',rotation=90,labelsize=8); ax.tick_params(axis='y',rotation=0,labelsize=8)
        else:
            cmap = plt.get_cmap("YlGnBu"); cmap.set_bad(color='lightgray'); im = ax.imshow(matrice,cmap=cmap,vmin=0,vmax=max_val,aspect='auto')
            ax.set_xticks(np.arange(len(all_numeri_sorted))); ax.set_yticks(np.arange(len(ruote_heatmap)))
            ax.set_xticklabels(all_numeri_sorted,rotation=90,fontsize=7); ax.set_yticklabels(ruote_heatmap,fontsize=7)
            for i_h in range(len(ruote_heatmap)):
                for j_h in range(len(all_numeri_sorted)):
                    val_h = matrice[i_h,j_h]
                    if not np.isnan(val_h): ax.text(j_h,i_h,f"{val_h:.1f}",ha="center",va="center",color="black" if val_h < 0.7*max_val else "white",fontsize=6)
            cbar = fig.colorbar(im,ax=ax,shrink=0.7); cbar.ax.set_ylabel(f"% {tipo_stat.capitalize()}",rotation=-90,va="bottom",fontsize=8); cbar.ax.tick_params(labelsize=7)
        plt.title(f"Heatmap {tipo_stat.capitalize()} ESTRATTI\n(Spia {numeri_spia_str} su {ruote_analisi_str})",fontsize=11,pad=15); plt.xlabel("Numeri Estratti",fontsize=9); plt.ylabel("Ruote Verifica",fontsize=9)
        plt.tight_layout(); return fig
    except Exception as e:
        print(f"Errore in crea_heatmap_correlazione: {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def visualizza_grafici(risultati_da_visualizzare, info_globale_ricerca, n_estrazioni_usate):
    try:
        if not risultati_da_visualizzare: messagebox.showinfo("Nessun Risultato", "Nessun risultato valido da visualizzare."); return
        win = tk.Toplevel(); win.title("Visualizzazione Grafica Risultati"); win.geometry("1300x850"); win.minsize(900,600)
        notebook = ttk.Notebook(win); notebook.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        def create_scrollable_tab(parent, tab_name):
            tab = ttk.Frame(parent); parent.add(tab, text=tab_name)
            canvas = tk.Canvas(tab); scrollbar_y = ttk.Scrollbar(tab,orient="vertical",command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(tab,orient="horizontal",command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0,0),window=scrollable_frame,anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_y.set,xscrollcommand=scrollbar_x.set)
            canvas.pack(side="top",fill="both",expand=True); scrollbar_y.pack(side="right",fill="y"); scrollbar_x.pack(side="bottom",fill="x"); return scrollable_frame
        barre_est_frame = create_scrollable_tab(notebook, "Grafici Barre (Estratti)")
        barre_amb_frame = create_scrollable_tab(notebook, "Grafici Barre (Ambi)")
        barre_ter_frame = create_scrollable_tab(notebook, "Grafici Barre (Terni)")
        tabelle_frame = create_scrollable_tab(notebook, "Tabelle Lotto (Solo Estratti)")
        heatmap_frame = create_scrollable_tab(notebook, "Heatmap Incrociata (Solo Estratti)")
        for frame, label_text in [(barre_est_frame, "Grafici a Barre per ESTRATTI Successivi"),
                                   (barre_amb_frame, "Grafici a Barre per AMBI Successivi"),
                                   (barre_ter_frame, "Grafici a Barre per TERNI Successivi"),
                                   (tabelle_frame, "Tabelle Lotto per ESTRATTI Successivi"),
                                   (heatmap_frame, "Heatmap Incrociata ESTRATTI tra Ruote di Verifica")]:
            ttk.Label(frame, text=label_text, style="Header.TLabel").pack(pady=10)
        for ruota_v, _, risultato in risultati_da_visualizzare:
            if risultato:
                info_specifica = info_globale_ricerca.copy(); info_specifica['ruota_verifica'] = ruota_v
                for tipo_an, frame_an in [("estratto", barre_est_frame), ("ambo", barre_amb_frame), ("terno", barre_ter_frame)]:
                    ruota_bar_frame_an = ttk.LabelFrame(frame_an, text=f"Ruota Verifica: {ruota_v} - {tipo_an.upper()}");
                    ruota_bar_frame_an.pack(fill="x",expand=False,padx=10,pady=10)
                    fig_p_an = crea_grafico_barre(risultato,info_specifica,tipo_an,"presenza")
                    fig_f_an = crea_grafico_barre(risultato,info_specifica,tipo_an,"frequenza")
                    if fig_p_an: FigureCanvasTkAgg(fig_p_an, master=ruota_bar_frame_an).get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                    if fig_f_an: FigureCanvasTkAgg(fig_f_an, master=ruota_bar_frame_an).get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
                    if not fig_p_an and not fig_f_an: ttk.Label(ruota_bar_frame_an,text=f"Nessun grafico {tipo_an} generato").pack(padx=5,pady=5)
                ruota_tab_frame = ttk.LabelFrame(tabelle_frame, text=f"Ruota Verifica: {ruota_v}"); ruota_tab_frame.pack(fill="x",expand=False,padx=10,pady=10)
                fig_tp = crea_tabella_lotto(risultato,info_specifica,"estratto","presenza"); fig_tf = crea_tabella_lotto(risultato,info_specifica,"estratto","frequenza")
                if fig_tp: FigureCanvasTkAgg(fig_tp, master=ruota_tab_frame).get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                if fig_tf: FigureCanvasTkAgg(fig_tf, master=ruota_tab_frame).get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
                if not fig_tp and not fig_tf: ttk.Label(ruota_tab_frame,text="Nessuna tabella generata").pack(padx=5,pady=5)
        risultati_validi_heatmap = [r for r in risultati_da_visualizzare if r[2] and isinstance(r[2],dict) and r[2].get('estratto')]
        if len(risultati_validi_heatmap) < 2: ttk.Label(heatmap_frame,text="Heatmap richiede >= 2 Ruote Verifica con risultati validi per gli estratti.").pack(padx=20,pady=20)
        else:
            for tipo_h_stat, nome_h_stat in [("presenza","Presenza"),("frequenza","Frequenza")]:
                heatmap_fig = crea_heatmap_correlazione(risultati_validi_heatmap,info_globale_ricerca,"estratto",tipo_h_stat)
                if heatmap_fig:
                    ttk.Label(heatmap_frame,text=f"--- Heatmap {nome_h_stat} (Estratti) ---",font=("Helvetica",11,"bold")).pack(pady=(10,5))
                    FigureCanvasTkAgg(heatmap_fig,master=heatmap_frame).get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)
                else: ttk.Label(heatmap_frame,text=f"Nessuna Heatmap {nome_h_stat} (Estratti) generata.").pack(pady=10)
    except Exception as e: messagebox.showerror("Errore Visualizzazione",f"Errore creazione finestra grafici:\n{e}"); traceback.print_exc()
    finally: plt.close('all')

# =============================================================================
# FUNZIONI LOGICHE
# =============================================================================
def carica_dati(source, start_date=None, end_date=None):
    lines = None
    try:
        if source.lower().startswith('http'):
            with urllib.request.urlopen(source) as response:
                if response.status != 200:
                    print(f"Errore HTTP {response.status} per URL: {source}")
                    return None
                lines = response.read().decode('utf-8').splitlines()
        else:
            if not os.path.exists(source):
                print(f"File locale non trovato: {source}")
                return None
            with open(source, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        if not lines: return None

        dates, ruote, numeri, seen_rows, fmt_ok = [], [], [], set(), '%Y/%m/%d'
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 7: continue
            data_str, ruota_str, nums_orig = parts[0], parts[1].upper(), parts[2:7]
            try:
                data_dt_val = datetime.datetime.strptime(data_str, fmt_ok)
                if len(nums_orig) != 5: continue
                [int(n) for n in nums_orig]
            except ValueError: continue

            if start_date and end_date and (data_dt_val.date() < start_date.date() or data_dt_val.date() > end_date.date()): continue
            key = f"{data_str}_{ruota_str}"
            if key in seen_rows: continue
            seen_rows.add(key)
            dates.append(data_str)
            ruote.append(ruota_str)
            numeri.append(nums_orig)

        if not dates: return None
        df = pd.DataFrame({'Data': dates, 'Ruota': ruote, **{f'Numero{i+1}': [n[i] for n in numeri] for i in range(5)}})
        df['Data'] = pd.to_datetime(df['Data'], format=fmt_ok)
        for col in [f'Numero{i+1}' for i in range(5)]:
             df[col] = df[col].apply(lambda x: str(int(x)).zfill(2) if pd.notna(x) and str(x).isdigit() and 1 <= int(x) <= 90 else pd.NA)
        df.dropna(subset=[f'Numero{i+1}' for i in range(5)], how='any', inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)
        return df if not df.empty else None

    except urllib.error.URLError as e:
        print(f"Errore di rete o URL per {source}: {e}")
        messagebox.showerror("Errore di Rete", f"Impossibile scaricare i dati.\nControlla la connessione internet.\n\nDettagli: {e}")
        return None
    except Exception as e:
        print(f"Errore durante l'elaborazione dei dati da {source}: {e}")
        traceback.print_exc()
        return None

def analizza_ruota_verifica(df_verifica, date_eventi_spia_sorted, n_estrazioni, nome_ruota_verifica):
    if df_verifica is None or df_verifica.empty: return None, "Df verifica vuoto."
    df_verifica = df_verifica.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    n_eventi_spia = len(date_eventi_spia_sorted)
    date_series_verifica = df_verifica['Data']

    freq_estratti, freq_ambi, freq_terne = Counter(), Counter(), Counter()
    pres_estratti, pres_ambi, pres_terne = Counter(), Counter(), Counter()

    freq_pos_estratti = {}
    pres_pos_estratti = {}

    for data_t in date_eventi_spia_sorted:
        try:
            start_index = date_series_verifica.searchsorted(data_t, side='right')
        except Exception:
            continue
        if start_index >= len(date_series_verifica):
            continue

        df_successive = df_verifica.iloc[start_index : start_index + n_estrazioni]
        estratti_unici_finestra, ambi_unici_finestra, terne_unici_finestra = set(), set(), set()
        estratti_pos_unici_finestra = {}

        if not df_successive.empty:
            for _, row in df_successive.iterrows():
                numeri_estratti_riga_con_pos = []
                for pos_idx, col_num_nome in enumerate(colonne_numeri):
                    num_val = row[col_num_nome]
                    if pd.notna(num_val):
                        numeri_estratti_riga_con_pos.append((str(num_val).zfill(2), pos_idx + 1))

                if not numeri_estratti_riga_con_pos: continue

                for num_str, pos in numeri_estratti_riga_con_pos:
                    freq_estratti[num_str] += 1
                    estratti_unici_finestra.add(num_str)

                    if num_str not in freq_pos_estratti:
                        freq_pos_estratti[num_str] = Counter()
                    freq_pos_estratti[num_str][pos] += 1

                    if num_str not in estratti_pos_unici_finestra:
                        estratti_pos_unici_finestra[num_str] = set()
                    estratti_pos_unici_finestra[num_str].add(pos)

                numeri_solo_per_combinazioni = sorted([item[0] for item in numeri_estratti_riga_con_pos])
                if len(numeri_solo_per_combinazioni) >= 2:
                    for ambo in itertools.combinations(numeri_solo_per_combinazioni, 2):
                        ambo_ordinato = tuple(sorted(ambo))
                        freq_ambi[ambo_ordinato] += 1
                        ambi_unici_finestra.add(ambo_ordinato)

                if len(numeri_solo_per_combinazioni) >= 3:
                    for terno in itertools.combinations(numeri_solo_per_combinazioni, 3):
                        terno_ordinato = tuple(sorted(terno))
                        freq_terne[terno_ordinato] += 1
                        terne_unici_finestra.add(terno_ordinato)

        for num in estratti_unici_finestra: pres_estratti[num] += 1
        for ambo_u in ambi_unici_finestra: pres_ambi[ambo_u] += 1
        for terno in terne_unici_finestra: pres_terne[terno] += 1

        for num_str, pos_set in estratti_pos_unici_finestra.items():
            if num_str not in pres_pos_estratti:
                pres_pos_estratti[num_str] = Counter()
            for pos_val in pos_set:
                pres_pos_estratti[num_str][pos_val] +=1

    results = {'totale_eventi_spia': n_eventi_spia}
    for tipo, freq_dict, pres_dict in [('estratto', freq_estratti, pres_estratti),
                                       ('ambo', freq_ambi, pres_ambi),
                                       ('terno', freq_terne, pres_terne)]:
        if not freq_dict:
            results[tipo] = None
            continue

        if tipo in ['ambo', 'terno']:
            freq_s = pd.Series({format_ambo_terno(k): v for k, v in freq_dict.items()}, dtype=int).sort_index()
            pres_s = pd.Series({format_ambo_terno(k): v for k, v in pres_dict.items()}, dtype=int)
        else:
            freq_s = pd.Series(freq_dict, dtype=int).sort_index()
            pres_s = pd.Series(pres_dict, dtype=int)

        pres_s = pres_s.reindex(freq_s.index, fill_value=0).sort_index()

        tot_freq = freq_s.sum()
        perc_freq = (freq_s / tot_freq * 100).round(2) if tot_freq > 0 else pd.Series(0.0, index=freq_s.index, dtype=float)
        perc_pres = (pres_s / n_eventi_spia * 100).round(2) if n_eventi_spia > 0 else pd.Series(0.0, index=pres_s.index, dtype=float)

        top_pres_items = pres_s.sort_values(ascending=False).head(10)
        top_freq_items = freq_s.sort_values(ascending=False).head(10)

        results[tipo] = {
            'presenza': {'top': top_pres_items,
                         'percentuali': perc_pres.reindex(top_pres_items.index).fillna(0.0),
                         'frequenze': freq_s.reindex(top_pres_items.index).fillna(0).astype(int),
                         'perc_frequenza': perc_freq.reindex(top_pres_items.index).fillna(0.0)},
            'frequenza': {'top': top_freq_items,
                          'percentuali': perc_freq.reindex(top_freq_items.index).fillna(0.0),
                          'presenze': pres_s.reindex(top_freq_items.index).fillna(0).astype(int),
                          'perc_presenza': perc_pres.reindex(top_freq_items.index).fillna(0.0)},
            'all_percentuali_presenza': perc_pres,
            'all_percentuali_frequenza': perc_freq,
            'full_presenze': pres_s,
            'full_frequenze': freq_s
        }

    if 'estratto' in results and results['estratto'] and freq_pos_estratti:
        results['estratto']['posizionale_frequenza'] = {
            num: dict(sorted(pos_counts.items())) for num, pos_counts in freq_pos_estratti.items()
        }
        results['estratto']['posizionale_presenza'] = {
            num: dict(sorted(pos_counts.items())) for num, pos_counts in pres_pos_estratti.items()
        }
    elif 'estratto' in results and results['estratto']:
        results['estratto']['posizionale_frequenza'] = {}
        results['estratto']['posizionale_presenza'] = {}

    return (results, None) if any(results.get(t) for t in ['estratto', 'ambo', 'terno']) else (None, f"Nessun risultato su {nome_ruota_verifica}.")

def analizza_antecedenti(df_ruota, numeri_obiettivo, n_precedenti, nome_ruota):
    if df_ruota is None or df_ruota.empty:
        return None, f"Nessun dato disponibile per la ruota {nome_ruota} nel periodo selezionato."

    if not numeri_obiettivo or n_precedenti <= 0:
        return None, "Input per l'analisi degli antecedenti non validi (numeri o colpi)."

    df_ruota = df_ruota.sort_values(by='Data').reset_index(drop=True)
    cols_num = [f'Numero{i+1}' for i in range(5)]
    numeri_obiettivo_zfill = [str(n).zfill(2) for n in numeri_obiettivo]

    indices_obiettivo = df_ruota.index[df_ruota[cols_num].isin(numeri_obiettivo_zfill).any(axis=1)].tolist()
    n_occ_obiettivo = len(indices_obiettivo)

    if n_occ_obiettivo == 0:
        return None, f"Nessuna occorrenza dei numeri obiettivo trovata sulla ruota {nome_ruota}."

    freq_ant, pres_ant = Counter(), Counter()
    actual_base_pres = 0

    for idx_obj in indices_obiettivo:
        if idx_obj < n_precedenti:
            continue
        actual_base_pres += 1
        df_prec = df_ruota.iloc[idx_obj - n_precedenti : idx_obj]
        if not df_prec.empty:
            numeri_finestra_unici = set()
            for _, row_prec in df_prec.iterrows():
                estratti_prec_riga = [row_prec[col] for col in cols_num if pd.notna(row_prec[col])]
                freq_ant.update(estratti_prec_riga)
                numeri_finestra_unici.update(estratti_prec_riga)
            pres_ant.update(list(numeri_finestra_unici))

    base_res = {
        'totale_occorrenze_obiettivo': n_occ_obiettivo,
        'base_presenza_antecedenti': actual_base_pres,
        'numeri_obiettivo': numeri_obiettivo_zfill,
        'n_precedenti': n_precedenti,
        'nome_ruota': nome_ruota
    }

    if actual_base_pres == 0:
        return base_res, f"Trovate {n_occ_obiettivo} occorrenze, ma nessuna aveva abbastanza estrazioni precedenti per l'analisi."

    if not freq_ant:
        return base_res, "Nessun numero antecedente trovato nelle finestre di analisi."

    ant_freq_s = pd.Series(freq_ant, dtype=int).sort_index()
    ant_pres_s = pd.Series(pres_ant, dtype=int).reindex(ant_freq_s.index, fill_value=0).sort_index()

    tot_ant_freq = ant_freq_s.sum()
    perc_ant_freq = (ant_freq_s / tot_ant_freq * 100).round(2) if tot_ant_freq > 0 else pd.Series(0.0, index=ant_freq_s.index)
    perc_ant_pres = (ant_pres_s / actual_base_pres * 100).round(2) if actual_base_pres > 0 else pd.Series(0.0, index=ant_pres_s.index)

    top_ant_pres = ant_pres_s.sort_values(ascending=False).head(10)
    top_ant_freq = ant_freq_s.sort_values(ascending=False).head(10)

    results_data = {
        **base_res,
        'presenza': {
            'top': top_ant_pres,
            'percentuali': perc_ant_pres.reindex(top_ant_pres.index).fillna(0.0),
            'frequenze': ant_freq_s.reindex(top_ant_pres.index).fillna(0).astype(int),
            'perc_frequenza': perc_ant_freq.reindex(top_ant_pres.index).fillna(0.0)
        },
        'frequenza': {
            'top': top_ant_freq,
            'percentuali': perc_ant_freq.reindex(top_ant_freq.index).fillna(0.0),
            'presenze': ant_pres_s.reindex(top_ant_freq.index).fillna(0).astype(int),
            'perc_presenza': perc_ant_pres.reindex(top_ant_freq.index).fillna(0.0)
        }
    }

    return results_data, None

def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
    global risultati_globali, info_ricerca_globale, button_visualizza, button_verifica_futura, button_verifica_mista

    if button_visualizza: button_visualizza.config(state=tk.DISABLED)
    if button_verifica_futura: button_verifica_futura.config(state=tk.DISABLED)
    if button_verifica_mista: button_verifica_mista.config(state=tk.DISABLED)

    if modalita == "successivi":
        risultati_globali = risultati_nuovi if risultati_nuovi is not None else []
        info_ricerca_globale = info_ricerca if info_ricerca is not None else {}
        has_valid_results = bool(risultati_globali) and any(res[2] for res in risultati_globali if len(res)>2)
        has_date_eventi_spia = bool(info_ricerca_globale.get('date_eventi_spia_ordinate'))
        has_end_date = info_ricerca_globale.get('end_date') is not None
        has_ruote_verifica_info = bool(info_ricerca_globale.get('ruote_verifica'))

        if has_valid_results and button_visualizza: button_visualizza.config(state=tk.NORMAL)
        if has_end_date and has_ruote_verifica_info and button_verifica_futura: button_verifica_futura.config(state=tk.NORMAL)
        if has_date_eventi_spia and has_ruote_verifica_info and button_verifica_mista: button_verifica_mista.config(state=tk.NORMAL)
    else:
        risultati_globali, info_ricerca_globale = [], {}

def salva_risultati():
    global risultato_text, root
    content = risultato_text.get(1.0, tk.END).strip()
    if not content or any(msg in content for msg in ["Benvenuto", "Ricerca in corso...", "Nessun risultato", "Elaborazione in corso..."]):
        messagebox.showinfo("Salvataggio", "Nessun risultato significativo da salvare.", parent=root)
        return
    fpath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Salva Risultati")
    if fpath:
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Salvataggio OK", f"Risultati salvati in:\n{fpath}", parent=root)
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e}", parent=root)

def format_ambo_terno(combinazione):
    if isinstance(combinazione, tuple) or isinstance(combinazione, list):
        return "-".join(map(str, combinazione))
    return str(combinazione)

def calcola_ritardo_attuale_globale(df_all_wheels, item_to_find, item_type, end_date):
    if df_all_wheels is None or df_all_wheels.empty:
        return "N/D"

    df_period = df_all_wheels[df_all_wheels['Data'] <= end_date]
    if df_period.empty:
        return "N/D"

    cols_num = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']

    hits_df = None
    if item_type == 'estratto':
        condition = (df_period[cols_num] == item_to_find).any(axis=1)
        hits_df = df_period[condition]
    elif item_type == 'ambo':
        n1, n2 = item_to_find
        # Ricerca vettorizzata: molto più veloce di .apply
        condition1 = (df_period[cols_num] == n1).any(axis=1)
        condition2 = (df_period[cols_num] == n2).any(axis=1)
        hits_df = df_period[condition1 & condition2]
    else:
        return "N/D"

    if not hits_df.empty:
        last_hit_date = hits_df['Data'].max()
        # Conta il numero di date di estrazione uniche dopo l'ultima uscita
        unique_dates_after_hit = df_period[df_period['Data'] > last_hit_date]['Data'].unique()
        return len(unique_dates_after_hit)
    else:
        # Se non è mai uscito, il ritardo è il numero totale di concorsi unici nel periodo
        return len(df_period['Data'].unique())

def calcola_ritardo_attuale(df_ruota_completa, item_da_cercare, tipo_item, data_fine_analisi):
    if df_ruota_completa is None or df_ruota_completa.empty:
        return "N/D (no data)"
    if not isinstance(data_fine_analisi, pd.Timestamp):
        data_fine_analisi = pd.Timestamp(data_fine_analisi)

    df_filtrato = df_ruota_completa[df_ruota_completa['Data'] <= data_fine_analisi].sort_values(by='Data', ascending=False)

    if df_filtrato.empty:
        return "N/D (no draws in period)"

    ritardo = 0
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']

    for _, row in df_filtrato.iterrows():
        ritardo += 1
        numeri_riga_set = {str(row[col]).zfill(2) for col in colonne_numeri if pd.notna(row[col])}

        trovato = False
        if tipo_item == "estratto":
            if isinstance(item_da_cercare, str) and item_da_cercare in numeri_riga_set:
                trovato = True
        elif tipo_item == "ambo":
            if isinstance(item_da_cercare, tuple) and len(item_da_cercare) == 2:
                item_ambo_set = set(str(n).zfill(2) for n in item_da_cercare)
                if item_ambo_set.issubset(numeri_riga_set):
                    trovato = True
        elif tipo_item == "terno":
            if isinstance(item_da_cercare, tuple) and len(item_da_cercare) == 3:
                item_terno_set = set(str(n).zfill(2) for n in item_da_cercare)
                if item_terno_set.issubset(numeri_riga_set):
                    trovato = True

        if trovato:
            return ritardo -1

    return ritardo

def calcola_copertura_ambo_combinazione(combinazione_tuple, ambi_copertura_dict):
    """
    Calcola la copertura totale degli ambi per una data combinazione (terno, quartina, ecc.).
    Una combinazione copre un evento se almeno uno dei suoi ambi interni è uscito.
    """
    if len(combinazione_tuple) < 2:
        return set()

    # Genera tutti gli ambi interni alla combinazione
    ambi_interni = itertools.combinations(combinazione_tuple, 2)

    # Unisce tutti i set di date in cui ogni ambo interno è uscito
    date_coperte = set()
    for ambo in ambi_interni:
        date_coperte.update(ambi_copertura_dict.get(tuple(sorted(ambo)), set()))

    return date_coperte

def calcola_ritardo_copertura_ambo(df_ruota_completa, combinazione_tuple, data_fine_analisi):
    """
    Calcola il ritardo di una combinazione (terzina, quartina, ecc.)
    inteso come il numero di estrazioni da cui non esce NESSUNO degli ambi che la compongono.
    """
    if df_ruota_completa is None or df_ruota_completa.empty or len(combinazione_tuple) < 2:
        return "N/D"
    if not isinstance(data_fine_analisi, pd.Timestamp):
        data_fine_analisi = pd.Timestamp(data_fine_analisi)

    df_filtrato = df_ruota_completa[df_ruota_completa['Data'] <= data_fine_analisi].sort_values(by='Data', ascending=False)
    if df_filtrato.empty:
        return "N/D"

    # Genera tutti gli ambi possibili dalla combinazione
    ambi_da_cercare = [set(map(str, ambo)) for ambo in itertools.combinations(combinazione_tuple, 2)]
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    ritardo = 0

    for _, row in df_filtrato.iterrows():
        numeri_riga_set = {str(row[col]).zfill(2) for col in colonne_numeri if pd.notna(row[col])}

        # Controlla se almeno uno degli ambi è presente
        trovato = any(ambo_set.issubset(numeri_riga_set) for ambo_set in ambi_da_cercare)

        if trovato:
            return ritardo  # Ritorna il ritardo al momento del ritrovamento

        ritardo += 1

    return ritardo # Ritorna il ritardo totale se non è mai uscito

def mostra_popup_testo_semplice(titolo, contenuto_testo):
    global root
    popup_window = tk.Toplevel(root)
    popup_window.title(titolo)

    num_righe = contenuto_testo.count('\n') + 1
    max_len_riga = 0
    if contenuto_testo:
        try: max_len_riga = max(len(r) for r in contenuto_testo.split('\n'))
        except ValueError: max_len_riga = 80
    else: max_len_riga = 80

    larghezza_stimata_char = max(80, min(120, max_len_riga + 5))
    altezza_stimata_righe = max(15, min(40, num_righe + 5))
    popup_width = larghezza_stimata_char * 7
    popup_height = altezza_stimata_righe * 16
    popup_width = max(500, min(900, popup_width))
    popup_height = max(300, min(700, popup_height))

    popup_window.geometry(f"{int(popup_width)}x{int(popup_height)}")
    popup_window.transient(root)

    text_widget = scrolledtext.ScrolledText(popup_window, wrap=tk.WORD, font=("Consolas", 10))
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    text_widget.insert(tk.END, contenuto_testo)
    text_widget.config(state=tk.DISABLED)

    button_frame_popup = ttk.Frame(popup_window)
    button_frame_popup.pack(fill=tk.X, pady=(0,10), padx=10, side=tk.BOTTOM)

    def _salva_popup_content_locale():
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Salva contenuto popup"
        )
        if fpath:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(contenuto_testo)
                messagebox.showinfo("Salvataggio OK", f"Contenuto del popup salvato in:\n{fpath}", parent=popup_window)
            except Exception as e_save:
                messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e_save}", parent=popup_window)

    ttk.Button(button_frame_popup, text="Salva su File...", command=_salva_popup_content_locale).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame_popup, text="Chiudi", command=popup_window.destroy).pack(side=tk.RIGHT, padx=5)

    popup_window.update_idletasks()
    master_x = root.winfo_x(); master_y = root.winfo_y()
    master_width = root.winfo_width(); master_height = root.winfo_height()
    win_width = popup_window.winfo_width(); win_height = popup_window.winfo_height()
    center_x = master_x + (master_width // 2) - (win_width // 2)
    center_y = master_y + (master_height // 2) - (win_height // 2)
    popup_window.geometry(f"+{center_x}+{center_y}")

def mostra_popup_risultati_spia(info_ricerca, risultati_analisi):
    global root

    popup = tk.Toplevel(root)
    popup.title("Riepilogo Analisi Numeri Spia")
    popup.geometry("950x800")
    popup.transient(root)

    text_area_popup = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
    text_area_popup.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10,0))

    popup_content_list = ["=== RIEPILOGO ANALISI NUMERI SPIA ===\n"]
    tipo_spia_usato = info_ricerca.get('tipo_spia_usato', 'N/D').upper()
    spia_val = info_ricerca.get('numeri_spia_input', [])
    spia_display = ""
    if tipo_spia_usato == "ESTRATTO_POSIZIONALE":
        num_spia_list = spia_val if isinstance(spia_val, list) else [spia_val]
        num_spia = num_spia_list[0] if num_spia_list else "N/D"
        pos_spia = info_ricerca.get('posizione_spia_input', "N/D")
        spia_display = f"{num_spia} in {pos_spia}a pos."
    elif isinstance(spia_val, tuple): spia_display = "-".join(map(str, spia_val))
    elif isinstance(spia_val, list): spia_display = ", ".join(map(str, spia_val))
    else: spia_display = str(spia_val)
    popup_content_list.append(f"Tipo Spia: {tipo_spia_usato.replace('_', ' ')} ({spia_display})")
    popup_content_list.append(f"Ruote Analisi (Spia): {', '.join(info_ricerca.get('ruote_analisi', []))}")
    popup_content_list.append(f"Ruote Verifica (Esiti): {', '.join(info_ricerca.get('ruote_verifica', []))}")
    start_date_pd = info_ricerca.get('start_date', pd.NaT); end_date_pd = info_ricerca.get('end_date', pd.NaT)
    start_date_str = start_date_pd.strftime('%d/%m/%Y') if pd.notna(start_date_pd) else "N/D"
    end_date_str = end_date_pd.strftime('%d/%m/%Y') if pd.notna(end_date_pd) else "N/D"
    popup_content_list.append(f"Periodo: {start_date_str} - {end_date_str}")
    popup_content_list.append(f"Estrazioni Successive Analizzate: {info_ricerca.get('n_estrazioni', 'N/D')}")
    date_eventi_spia = info_ricerca.get('date_eventi_spia_ordinate', [])
    popup_content_list.append(f"Numero Totale di Eventi Spia: {len(date_eventi_spia)}")
    
    # --- INIZIO BLOCCO RISULTATI PER RUOTA ---
    if risultati_analisi:
        popup_content_list.append("\n" + "=" * 10 + " ANALISI PER RUOTA DI VERIFICA " + "="*10)
        for nome_ruota_v, _, res_ruota in risultati_analisi:
            if not res_ruota or not isinstance(res_ruota, dict): continue
            popup_content_list.append(f"\n--- RISULTATI PER RUOTA: {nome_ruota_v.upper()} ---")
            tot_eventi_spia_analizzati = res_ruota.get('totale_eventi_spia', 0)
            popup_content_list.append(f"(Basato su {tot_eventi_spia_analizzati} eventi spia analizzabili)")
            for tipo_esito in ['estratto', 'ambo', 'terno']:
                dati_esito = res_ruota.get(tipo_esito)
                if dati_esito:
                    popup_content_list.append(f"\n  -- {tipo_esito.capitalize()} Successivi (Top per Presenza) --")
                    top_pres_items = dati_esito.get('presenza', {}).get('top')
                    if top_pres_items is not None and not top_pres_items.empty:
                        for item_str, pres_val in top_pres_items.items():
                            perc = dati_esito['presenza']['percentuali'].get(item_str, 0.0)
                            freq = dati_esito['presenza']['frequenze'].get(item_str, 0)
                            
                            riga = f"    - {item_str}: Pres. {pres_val} su {tot_eventi_spia_analizzati} eventi ({perc:.1f}%) | Freq.Tot: {freq}"
                            
                            if tipo_esito in ['estratto', 'ambo'] and 'ritardi_attuali' in dati_esito:
                                ritardo_val = dati_esito['ritardi_attuali'].get(item_str, "N/D")
                                riga += f" | Rit.Att: {ritardo_val}"
                            
                            popup_content_list.append(riga)

                            if tipo_esito == 'estratto' and 'posizionale_presenza' in dati_esito:
                                pos_data = dati_esito['posizionale_presenza'].get(item_str)
                                if pos_data:
                                    pos_str_list = [f"P{p}:{c}({(c/pres_val*100):.0f}%)" for p, c in pos_data.items()]
                                    if pos_str_list:
                                        popup_content_list.append(f"      Posizioni (Pres.): {', '.join(pos_str_list)}")
                    else: popup_content_list.append("    Nessuno.")
    
    # --- FINE BLOCCO RISULTATI PER RUOTA ---


    # --- INIZIO BLOCCO RISULTATI GLOBALI ---
    n_eventi_spia_tot = len(date_eventi_spia)

    # Estratti
    if info_ricerca.get('migliori_estratti_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI ESTRATTI PER COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_estratti_copertura_globale']:
            rit_str = f" | Rit.Att: {item['ritardo_attuale']}" if item['ritardo_attuale'] != 'N/C' else ''
            popup_content_list.append(f"  - Estratto {item['estratto']}: Coperti {item['coperti']} su {item['totali']} eventi ({item['percentuale']:.1f}%)" + rit_str)
        if 'summary_estratti_globali' in info_ricerca:
             summary_info = info_ricerca['summary_estratti_globali']
             popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando questi {summary_info['giocati']} estratti si coprono {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")

    # Ambi
    if info_ricerca.get('migliori_ambi_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI AMBI PER COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_ambi_copertura_globale']:
            rit_str = f" | Rit.Att: {item['ritardo_attuale']}" if item['ritardo_attuale'] != 'N/C' else ''
            popup_content_list.append(f"  - Ambo {item['ambo']}: Coperti {item['coperti']} su {item['totali']} eventi ({item['percentuale']:.1f}%)" + rit_str)
        if 'summary_ambi_globali' in info_ricerca:
            summary_info = info_ricerca['summary_ambi_globali']
            popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando questi {summary_info['giocati']} ambi si coprono {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")

    # Terni
    if info_ricerca.get('migliori_terni_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI TERNI (SECCHI) PER COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_terni_copertura_globale']:
            popup_content_list.append(f"  - Terno {item['terno']}: Coperti {item['coperti']} su {item['totali']} eventi ({item['percentuale']:.1f}%)")
        if 'summary_terni_globali' in info_ricerca:
            summary_info = info_ricerca['summary_terni_globali']
            popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando questi {summary_info['giocati']} terni si coprono {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")

    # Terni per Ambo
    if info_ricerca.get('migliori_terni_per_ambo_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI TERNI PER AMBO A COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_terni_per_ambo_copertura_globale']:
            perc = (item['copertura_ambo'] / n_eventi_spia_tot * 100) if n_eventi_spia_tot > 0 else 0
            popup_content_list.append(f"  - Terno {format_ambo_terno(item['combinazione'])}: Copre {item['copertura_ambo']} eventi per AMBO ({perc:.1f}%)")
        if 'summary_terni_per_ambo_globali' in info_ricerca:
            summary_info = info_ricerca['summary_terni_per_ambo_globali']
            popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando questi {summary_info['giocati']} terni si ottiene una vincita di AMBO in {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")

    # Quartine per Ambo
    if info_ricerca.get('migliori_quartine_per_ambo_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI QUARTINE PER AMBO A COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_quartine_per_ambo_copertura_globale']:
            perc = (item['copertura_ambo'] / n_eventi_spia_tot * 100) if n_eventi_spia_tot > 0 else 0
            popup_content_list.append(f"  - Quartina {format_ambo_terno(item['combinazione'])}: Copre {item['copertura_ambo']} eventi per AMBO ({perc:.1f}%)")
        if 'summary_quartine_per_ambo_globali' in info_ricerca:
            summary_info = info_ricerca['summary_quartine_per_ambo_globali']
            popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando queste {summary_info['giocati']} quartine si ottiene una vincita di AMBO in {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")
            
    # Cinquine per Ambo
    if info_ricerca.get('migliori_cinquine_per_ambo_copertura_globale'):
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI CINQUINE PER AMBO A COPERTURA GLOBALE " + "=" * 10)
        for item in info_ricerca['migliori_cinquine_per_ambo_copertura_globale']:
            perc = (item['copertura_ambo'] / n_eventi_spia_tot * 100) if n_eventi_spia_tot > 0 else 0
            popup_content_list.append(f"  - Cinquina {format_ambo_terno(item['combinazione'])}: Copre {item['copertura_ambo']} eventi per AMBO ({perc:.1f}%)")
        if 'summary_cinquine_per_ambo_globali' in info_ricerca:
            summary_info = info_ricerca['summary_cinquine_per_ambo_globali']
            popup_content_list.append(f"\n    RIEPILOGO COMBINATO: Giocando queste {summary_info['giocati']} cinquine si ottiene una vincita di AMBO in {summary_info['coperti']} su {n_eventi_spia_tot} eventi ({summary_info['percentuale']:.1f}%).")

    final_popup_text_content = "\n".join(popup_content_list)
    text_area_popup.config(state=tk.NORMAL)
    text_area_popup.delete(1.0, tk.END)
    text_area_popup.insert(tk.END, final_popup_text_content)
    text_area_popup.config(state=tk.DISABLED)

    button_frame_popup_spia = ttk.Frame(popup)
    button_frame_popup_spia.pack(fill=tk.X, pady=(5,10), padx=10, side=tk.BOTTOM)

    def _salva_popup_spia_content_definitiva():
        fpath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")], title="Salva Riepilogo Analisi Spia")
        if fpath:
            try:
                with open(fpath, "w", encoding="utf-8") as f: f.write(final_popup_text_content)
                messagebox.showinfo("Salvataggio OK", f"Riepilogo salvato in:\n{fpath}", parent=popup)
            except Exception as e_save:
                messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e_save}", parent=popup)

    ttk.Button(button_frame_popup_spia, text="Salva su File...", command=_salva_popup_spia_content_definitiva).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame_popup_spia, text="Chiudi", command=popup.destroy).pack(side=tk.RIGHT, padx=5)

    popup.update_idletasks()
    master_x = root.winfo_x(); master_y = root.winfo_y()
    master_width = root.winfo_width(); master_height = root.winfo_height()
    win_width = popup.winfo_width(); win_height = popup.winfo_height()
    center_x = master_x + (master_width // 2) - (win_width // 2)
    center_y = master_y + (master_height // 2) - (win_height // 2)
    popup.geometry(f"+{center_x}+{center_y}")

def cerca_numeri(modalita="successivi"):
    global risultati_globali, info_ricerca_globale, URL_RUOTE, file_ruote, risultato_text, root, data_source_var
    global start_date_entry, end_date_entry, listbox_ruote_analisi, listbox_ruote_verifica
    global entry_numeri_spia, estrazioni_entry_succ, listbox_ruote_analisi_ant
    global entry_numeri_obiettivo, estrazioni_entry_ant, tipo_spia_var_global, combo_posizione_spia
    global MAX_COLPI_GIOCO
    global entry_num_estratti_popup, entry_num_ambi_popup, entry_num_terni_popup, entry_num_terni_per_ambo_popup
    global entry_num_quartine_per_ambo_popup, entry_num_cinquine_per_ambo_popup
    global calcola_coperture_var, calcola_ritardi_var

    if data_source_var.get() == "Locale" and (not mappa_file_ruote() or not file_ruote):
        messagebox.showerror("Errore Cartella", "Modalità 'Locale' selezionata, ma impossibile leggere i file dalla cartella specificata o la cartella non contiene file validi.\nAssicurati che il percorso sia corretto e la cartella contenga file TXT delle ruote.")
        return

    risultati_globali,info_ricerca_globale = [],{}
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END,f"Ricerca {modalita} in corso...\nAttendere prego.\n")
    risultato_text.config(state=tk.DISABLED)
    root.update_idletasks()

    aggiorna_risultati_globali([],{},modalita=modalita)

    try:
        start_dt,end_dt = start_date_entry.get_date(),end_date_entry.get_date()
        if start_dt > end_dt: raise ValueError("Data inizio dopo data fine.")
        start_ts,end_ts = pd.Timestamp(start_dt),pd.Timestamp(end_dt)
    except Exception as e:
        messagebox.showerror("Input Date",f"Date non valide: {e}")
        return

    messaggi_out,ris_graf_loc = [],[]
    col_num_nomi = [f'Numero{i+1}' for i in range(5)]
    is_online = data_source_var.get() == "Online"

    if modalita == "successivi":
        esegui_calcolo_coperture = calcola_coperture_var.get()
        esegui_calcolo_ritardi = calcola_ritardi_var.get()

        try:
            num_estratti_da_mostrare = int(entry_num_estratti_popup.get())
            if not 1 <= num_estratti_da_mostrare <= 20: raise ValueError()
        except: num_estratti_da_mostrare = 10 
        try:
            num_ambi_da_mostrare = int(entry_num_ambi_popup.get())
            if not 1 <= num_ambi_da_mostrare <= 20: raise ValueError()
        except: num_ambi_da_mostrare = 10
        try:
            num_terni_da_mostrare = int(entry_num_terni_popup.get())
            if not 1 <= num_terni_da_mostrare <= 20: raise ValueError()
        except: num_terni_da_mostrare = 10
        try:
            num_terni_per_ambo_da_mostrare = int(entry_num_terni_per_ambo_popup.get())
            if not 1 <= num_terni_per_ambo_da_mostrare <= 20: raise ValueError()
        except: num_terni_per_ambo_da_mostrare = 3
        try:
            num_quartine_per_ambo_da_mostrare = int(entry_num_quartine_per_ambo_popup.get())
            if not 1 <= num_quartine_per_ambo_da_mostrare <= 20: raise ValueError()
        except: num_quartine_per_ambo_da_mostrare = 3
        try:
            num_cinquine_per_ambo_da_mostrare = int(entry_num_cinquine_per_ambo_popup.get())
            if not 1 <= num_cinquine_per_ambo_da_mostrare <= 20: raise ValueError()
        except: num_cinquine_per_ambo_da_mostrare = 3

        ra_idx,rv_idx = listbox_ruote_analisi.curselection(),listbox_ruote_verifica.curselection()
        if not ra_idx or not rv_idx:
            messagebox.showwarning("Manca Input","Seleziona Ruote Analisi (Spia) e Ruote Verifica (Esiti)."); return
        nomi_ra,nomi_rv = [listbox_ruote_analisi.get(i) for i in ra_idx],[listbox_ruote_verifica.get(i) for i in rv_idx]
        tipo_spia_scelto = tipo_spia_var_global.get() if tipo_spia_var_global else "estratto"
        numeri_spia_input_raw = [e.get().strip() for e in entry_numeri_spia]
        numeri_spia_input_validi_zfill = [str(int(n_str)).zfill(2) for n_str in numeri_spia_input_raw if n_str.isdigit() and 1 <= int(n_str) <= 90]
        numeri_spia_da_usare = None; spia_display_str = ""; posizione_spia_selezionata = None
        if tipo_spia_scelto == "estratto":
            if not numeri_spia_input_validi_zfill: messagebox.showwarning("Manca Input","Nessun Numero Spia (Estratto) valido.");return
            numeri_spia_da_usare = numeri_spia_input_validi_zfill; spia_display_str = ", ".join(numeri_spia_da_usare)
        elif tipo_spia_scelto == "estratto_posizionale":
            if not numeri_spia_input_validi_zfill or not numeri_spia_input_raw[0].strip(): messagebox.showwarning("Manca Input","Inserire il primo Numero Spia per l'analisi posizionale.");return
            primo_numero_spia = numeri_spia_input_validi_zfill[0]; numeri_spia_da_usare = [primo_numero_spia]; posizione_scelta_str = combo_posizione_spia.get()
            if not posizione_scelta_str or "Qualsiasi" in posizione_scelta_str: tipo_spia_scelto = "estratto"; spia_display_str = primo_numero_spia
            else:
                try: posizione_spia_selezionata = int(posizione_scelta_str.split("a")[0]); assert 1 <= posizione_spia_selezionata <= 5; spia_display_str = f"{primo_numero_spia} in {posizione_spia_selezionata}a pos."
                except: messagebox.showerror("Input Invalido",f"Posizione Spia non valida.");return
        elif tipo_spia_scelto == "ambo":
            if len(numeri_spia_input_validi_zfill) < 2: messagebox.showwarning("Manca Input","Inserire almeno 2 numeri validi per Ambo Spia.");return
            numeri_spia_da_usare = tuple(sorted(numeri_spia_input_validi_zfill[:2])); spia_display_str = "-".join(numeri_spia_da_usare)
        else: messagebox.showerror("Errore Interno", f"Tipo spia '{tipo_spia_scelto}' non gestito.");return
        
        try: 
            n_estr=int(estrazioni_entry_succ.get())
            assert 1 <= n_estr <= MAX_COLPI_GIOCO
        except: 
            messagebox.showerror("Input Invalido",f"N. Estrazioni (1-{MAX_COLPI_GIOCO}) non valido.");return
        
        info_curr={'numeri_spia_input':numeri_spia_da_usare,'tipo_spia_usato': tipo_spia_scelto,'ruote_analisi':nomi_ra,'ruote_verifica':nomi_rv,'n_estrazioni':n_estr,'start_date':start_ts,'end_date':end_ts}
        if posizione_spia_selezionata is not None and tipo_spia_scelto == "estratto_posizionale": info_curr['posizione_spia_input'] = posizione_spia_selezionata
        all_date_eventi_spia=set(); messaggi_out.append("\n--- FASE 1: Ricerca Date Uscita Spia ---")
        for nome_ra_loop in nomi_ra:
            source = URL_RUOTE.get(nome_ra_loop.upper()) if is_online else file_ruote.get(nome_ra_loop.upper())
            df_an=carica_dati(source, start_ts, end_ts)
            if df_an is None or df_an.empty: messaggi_out.append(f"[{nome_ra_loop}] No dati An."); continue
            dates_found_this_ruota = []
            if tipo_spia_scelto == "estratto": dates_found_arr=df_an.loc[df_an[col_num_nomi].isin(numeri_spia_da_usare).any(axis=1),'Data'].unique(); dates_found_this_ruota = pd.to_datetime(dates_found_arr).tolist()
            elif tipo_spia_scelto == "estratto_posizionale" and posizione_spia_selezionata is not None:
                numero_spia_pos = numeri_spia_da_usare[0]; colonna_target_pos = f'Numero{posizione_spia_selezionata}'
                if colonna_target_pos in df_an.columns: dates_found_arr = df_an.loc[df_an[colonna_target_pos] == numero_spia_pos, 'Data'].unique(); dates_found_this_ruota = pd.to_datetime(dates_found_arr).tolist()
            elif tipo_spia_scelto == "ambo":
                spia_n1, spia_n2 = numeri_spia_da_usare[0], numeri_spia_da_usare[1]
                presente_n1 = df_an[col_num_nomi].isin([spia_n1]).any(axis=1); presente_n2 = df_an[col_num_nomi].isin([spia_n2]).any(axis=1)
                date_ambo_presente = df_an.loc[presente_n1 & presente_n2, 'Data'].unique(); dates_found_this_ruota = pd.to_datetime(date_ambo_presente).tolist()
            if dates_found_this_ruota: all_date_eventi_spia.update(dates_found_this_ruota); messaggi_out.append(f"[{nome_ra_loop}] Trovate {len(dates_found_this_ruota)} date per Evento Spia {spia_display_str}.")
            else: messaggi_out.append(f"[{nome_ra_loop}] Nessuna uscita spia {spia_display_str}.")

        if not all_date_eventi_spia:
            messaggi_out.append(f"\nNESSUNA USCITA SPIA TROVATA PER {spia_display_str}.")
            aggiorna_risultati_globali([],info_curr,modalita="successivi")
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali); return
        date_eventi_spia_ord=sorted(list(all_date_eventi_spia)); n_eventi_spia_tot=len(date_eventi_spia_ord)
        info_curr['date_eventi_spia_ordinate']=date_eventi_spia_ord

        messaggi_out.append("\n--- FASE 2: Analisi Ruote Verifica ---")
        df_cache_completi_per_ritardo = {}
        
        if esegui_calcolo_coperture or esegui_calcolo_ritardi:
            for nome_rv_loop in nomi_rv:
                if nome_rv_loop not in df_cache_completi_per_ritardo:
                    source_full = URL_RUOTE.get(nome_rv_loop.upper()) if is_online else file_ruote.get(nome_rv_loop.upper())
                    df_full = carica_dati(source_full, start_date=None, end_date=None)
                    if df_full is not None: df_cache_completi_per_ritardo[nome_rv_loop] = df_full

        for nome_rv_loop in nomi_rv:
            source_ver = URL_RUOTE.get(nome_rv_loop.upper()) if is_online else file_ruote.get(nome_rv_loop.upper())
            df_ver_per_analisi_spia = carica_dati(source_ver, start_date=start_ts, end_date=None)
            if df_ver_per_analisi_spia is None or df_ver_per_analisi_spia.empty: continue
            res_ver, err_ver = analizza_ruota_verifica(df_ver_per_analisi_spia, date_eventi_spia_ord, n_estr, nome_rv_loop)
            if err_ver: continue
            if res_ver:
                df_storico_ruota_rit = df_cache_completi_per_ritardo.get(nome_rv_loop)
                if esegui_calcolo_ritardi and df_storico_ruota_rit is not None:
                    for tipo_ritardo in ['estratto', 'ambo']:
                        if tipo_ritardo in res_ver and res_ver[tipo_ritardo]:
                            res_ver[tipo_ritardo]['ritardi_attuali'] = {}
                            items_to_check = set(res_ver[tipo_ritardo]['presenza']['top'].index)
                            for item_str in items_to_check:
                                item_tuple = tuple(item_str.split('-')) if tipo_ritardo == 'ambo' else item_str
                                ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, item_tuple, tipo_ritardo, end_ts)
                                res_ver[tipo_ritardo]['ritardi_attuali'][item_str] = ritardo_val
                ris_graf_loc.append((nome_rv_loop, spia_display_str, res_ver))

        if esegui_calcolo_coperture and n_eventi_spia_tot > 0:
            df_completo_concatenato = pd.concat(df_cache_completi_per_ritardo.values()) if esegui_calcolo_ritardi and df_cache_completi_per_ritardo else None

            estratti_copertura, ambi_copertura, terni_copertura, quartine_copertura, cinquine_copertura = ({} for _ in range(5))

            for data_evento in date_eventi_spia_ord:
                e_usciti, a_usciti, t_usciti, q_usciti, c_usciti = (set() for _ in range(5))
                for nome_rv in nomi_rv:
                    df_ver = df_cache_completi_per_ritardo.get(nome_rv)
                    if df_ver is None: continue
                    start_idx = df_ver['Data'].searchsorted(data_evento, side='right')
                    df_successive = df_ver.iloc[start_idx : start_idx + n_estr]
                    if not df_successive.empty:
                        for _, row in df_successive.iterrows():
                            numeri_riga = sorted([str(row[col]) for col in col_num_nomi if pd.notna(row[col])])
                            e_usciti.update(numeri_riga)
                            if len(numeri_riga) >= 2: a_usciti.update(itertools.combinations(numeri_riga, 2))
                            if len(numeri_riga) >= 3: t_usciti.update(itertools.combinations(numeri_riga, 3))
                            if len(numeri_riga) >= 4: q_usciti.update(itertools.combinations(numeri_riga, 4))
                            if len(numeri_riga) >= 5: c_usciti.update(itertools.combinations(numeri_riga, 5))
                
                for e in e_usciti: estratti_copertura.setdefault(e, set()).add(data_evento)
                for a in a_usciti: ambi_copertura.setdefault(a, set()).add(data_evento)
                for t in t_usciti: terni_copertura.setdefault(t, set()).add(data_evento)
                for q in q_usciti: quartine_copertura.setdefault(q, set()).add(data_evento)
                for c in c_usciti: cinquine_copertura.setdefault(c, set()).add(data_evento)

            # ESTRATTI GLOBALI
            info_curr['migliori_estratti_copertura_globale'] = []
            if estratti_copertura:
                conteggio = Counter({e: len(d) for e, d in estratti_copertura.items()})
                migliori_raw = sorted(conteggio.items(), key=lambda item: (item[1], int(item[0])), reverse=True)[:num_estratti_da_mostrare]
                for estratto_str, count in migliori_raw:
                    rit_att = calcola_ritardo_attuale_globale(df_completo_concatenato, estratto_str, "estratto", end_ts) if esegui_calcolo_ritardi else "N/C"
                    info_curr['migliori_estratti_copertura_globale'].append({"estratto": estratto_str, "coperti": count, "totali": n_eventi_spia_tot, "percentuale": (count/n_eventi_spia_tot*100), "ritardo_attuale": rit_att})
                eventi_coperti = set().union(*(estratti_copertura.get(e, set()) for e, _ in migliori_raw))
                info_curr['summary_estratti_globali'] = {'giocati': len(migliori_raw), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}

            # AMBI GLOBALI
            info_curr['migliori_ambi_copertura_globale'] = []
            if ambi_copertura:
                conteggio = Counter({a: len(d) for a, d in ambi_copertura.items()})
                migliori_raw = sorted(conteggio.items(), key=lambda item: (item[1], int(item[0][0]), int(item[0][1])), reverse=True)[:num_ambi_da_mostrare]
                for ambo_tuple, count in migliori_raw:
                    rit_att = calcola_ritardo_attuale_globale(df_completo_concatenato, ambo_tuple, "ambo", end_ts) if esegui_calcolo_ritardi else "N/C"
                    info_curr['migliori_ambi_copertura_globale'].append({"ambo": format_ambo_terno(ambo_tuple), "coperti": count, "totali": n_eventi_spia_tot, "percentuale": (count/n_eventi_spia_tot*100), "ritardo_attuale": rit_att})
                eventi_coperti = set().union(*(ambi_copertura.get(a, set()) for a, _ in migliori_raw))
                info_curr['summary_ambi_globali'] = {'giocati': len(migliori_raw), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}
            
            # TERNI GLOBALI
            info_curr['migliori_terni_copertura_globale'] = []
            if terni_copertura:
                conteggio = Counter({t: len(d) for t, d in terni_copertura.items()})
                migliori_raw = sorted(conteggio.items(), key=lambda item: (item[1], int(item[0][0])), reverse=True)[:num_terni_da_mostrare]
                for terno_tuple, count in migliori_raw:
                    info_curr['migliori_terni_copertura_globale'].append({"terno": format_ambo_terno(terno_tuple), "coperti": count, "totali": n_eventi_spia_tot, "percentuale": (count/n_eventi_spia_tot*100)})
                eventi_coperti = set().union(*(terni_copertura.get(t, set()) for t, _ in migliori_raw))
                info_curr['summary_terni_globali'] = {'giocati': len(migliori_raw), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}
            
            # TERNI PER AMBO
            info_curr['migliori_terni_per_ambo_copertura_globale'] = []
            if terni_copertura:
                lista = [{'combinazione': t, 'copertura_ambo': len(calcola_copertura_ambo_combinazione(t, ambi_copertura))} for t in terni_copertura.keys()]
                top_items_unique = sorted(lista, key=lambda x: x['copertura_ambo'], reverse=True)[:num_terni_per_ambo_da_mostrare]
                info_curr['migliori_terni_per_ambo_copertura_globale'] = top_items_unique
                eventi_coperti = set().union(*(calcola_copertura_ambo_combinazione(i['combinazione'], ambi_copertura) for i in top_items_unique))
                info_curr['summary_terni_per_ambo_globali'] = {'giocati': len(top_items_unique), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}

            # QUARTINE PER AMBO
            info_curr['migliori_quartine_per_ambo_copertura_globale'] = []
            if quartine_copertura:
                lista = [{'combinazione': q, 'copertura_ambo': len(calcola_copertura_ambo_combinazione(q, ambi_copertura))} for q in quartine_copertura.keys()]
                top_items_unique = sorted(lista, key=lambda x: x['copertura_ambo'], reverse=True)[:num_quartine_per_ambo_da_mostrare]
                info_curr['migliori_quartine_per_ambo_copertura_globale'] = top_items_unique
                eventi_coperti = set().union(*(calcola_copertura_ambo_combinazione(i['combinazione'], ambi_copertura) for i in top_items_unique))
                info_curr['summary_quartine_per_ambo_globali'] = {'giocati': len(top_items_unique), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}
            
            # CINQUINE PER AMBO
            info_curr['migliori_cinquine_per_ambo_copertura_globale'] = []
            if cinquine_copertura:
                lista = [{'combinazione': c, 'copertura_ambo': len(calcola_copertura_ambo_combinazione(c, ambi_copertura))} for c in cinquine_copertura.keys()]
                top_items_unique = sorted(lista, key=lambda x: x['copertura_ambo'], reverse=True)[:num_cinquine_per_ambo_da_mostrare]
                info_curr['migliori_cinquine_per_ambo_copertura_globale'] = top_items_unique
                eventi_coperti = set().union(*(calcola_copertura_ambo_combinazione(i['combinazione'], ambi_copertura) for i in top_items_unique))
                info_curr['summary_cinquine_per_ambo_globali'] = {'giocati': len(top_items_unique), 'coperti': len(eventi_coperti), 'percentuale': (len(eventi_coperti)/n_eventi_spia_tot*100)}
        
        aggiorna_risultati_globali(ris_graf_loc,info_curr,modalita="successivi")
        mostra_popup_risultati_spia(info_curr, ris_graf_loc)

    elif modalita == "antecedenti":
        # ... (Questa parte rimane invariata rispetto alla tua versione)
        ra_ant_idx=listbox_ruote_analisi_ant.curselection()
        if not ra_ant_idx:
            messagebox.showwarning("Manca Input","Seleziona Ruota/e Analisi."); return
        nomi_ra_ant=[listbox_ruote_analisi_ant.get(i) for i in ra_ant_idx]
        num_obj_raw = [e.get().strip() for e in entry_numeri_obiettivo if e.get().strip() and e.get().strip().isdigit() and 1<=int(e.get().strip())<=90]
        num_obj = sorted(list(set(str(int(n)).zfill(2) for n in num_obj_raw)))
        if not num_obj:
            messagebox.showwarning("Manca Input","Nessun Numero Obiettivo (1-90) valido inserito."); return
        try:
            n_prec=int(estrazioni_entry_ant.get()); assert n_prec >=1
        except:
            messagebox.showerror("Input Invalido","N. Estrazioni Precedenti (>=1) non valido."); return
        messaggi_out.append(f"--- Analisi Antecedenti (Marker) ---")
        messaggi_out.append(f"Numeri Obiettivo: {', '.join(num_obj)}")
        messaggi_out.append(f"Numero Estrazioni Precedenti Controllate: {n_prec}")
        messaggi_out.append(f"Periodo: {start_ts.strftime('%d/%m/%Y')} - {end_ts.strftime('%d/%m/%Y')}")
        messaggi_out.append("-" * 40)
        df_cache_ant={}; almeno_un_risultato_significativo = False
        for nome_ra_ant_loop in nomi_ra_ant:
            df_ant_full = df_cache_ant.get(nome_ra_ant_loop)
            if df_ant_full is None: 
                source_ant = URL_RUOTE.get(nome_ra_ant_loop.upper()) if is_online else file_ruote.get(nome_ra_ant_loop.upper())
                df_ant_full=carica_dati(source_ant,start_ts,end_ts)
                df_cache_ant[nome_ra_ant_loop]=df_ant_full
            if df_ant_full is None or df_ant_full.empty:
                messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato storico trovato per il periodo selezionato."); continue
            res_ant, err_ant = analizza_antecedenti(df_ruota=df_ant_full, numeri_obiettivo=num_obj, n_precedenti=n_prec, nome_ruota=nome_ra_ant_loop)
            if res_ant is None and err_ant:
                messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Errore: {err_ant}"); continue
            if res_ant:
                if err_ant:
                     messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Info: {err_ant}")
                elif res_ant.get('presenza') and not res_ant['presenza']['top'].empty:
                    almeno_un_risultato_significativo = True
                    msg_res_ant=f"\n=== Risultati Antecedenti per Ruota: {nome_ra_ant_loop.upper()} ==="
                    msg_res_ant+=f"\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Estrazioni Prec.: {res_ant['n_precedenti']} | Occorrenze Obiettivo: {res_ant['totale_occorrenze_obiettivo']})"
                    msg_res_ant+=f"\n  Top Antecedenti per Presenza (su {res_ant['base_presenza_antecedenti']} casi validi):"
                    for i,(num,pres) in enumerate(res_ant['presenza']['top'].head(10).items()):
                        perc_pres_val = res_ant['presenza']['percentuali'].get(num,0.0); freq_val = res_ant['presenza']['frequenze'].get(num,0); 
                        msg_res_ant+=f"\n    {i+1}. {num}: {pres} ({perc_pres_val:.1f}%) [Freq.Tot: {freq_val}]"
                    messaggi_out.append(msg_res_ant)
                else:
                    messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato antecedente significativo trovato per i numeri obiettivo specificati.")
            messaggi_out.append("\n" + ("- "*20))
        aggiorna_risultati_globali([],{},modalita="antecedenti")
        if not almeno_un_risultato_significativo and any("Nessun dato" in m or "Errore" in m or "Info" in m for m in messaggi_out):
             pass
        elif not almeno_un_risultato_significativo:
             messaggi_out.append("\n\nRICERCA COMPLETATA: Nessun risultato significativo trovato per i criteri inseriti su nessuna delle ruote selezionate.")

    final_output="\n".join(messaggi_out) if messaggi_out else "Nessun risultato."
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END,final_output)
    risultato_text.see("1.0")
    
    if modalita == 'antecedenti':
        mostra_popup_testo_semplice("Riepilogo Analisi Numeri Antecedenti", final_output)

def verifica_esiti_utente_su_eventi_spia(date_eventi_spia, combinazioni_utente, nomi_ruote_verifica, n_verifiche, start_ts, end_ts, titolo_sezione="VERIFICA MISTA SU EVENTI SPIA"):
    if not date_eventi_spia or not combinazioni_utente or not nomi_ruote_verifica:
        return "Errore: Dati input mancanti per verifica utente su eventi spia."
    estratti_u, ambi_u_tpl, terni_u_tpl, quaterne_u_tpl, cinquine_u_tpl = [], [], [], [], []
    if isinstance(combinazioni_utente, dict):
        estratti_u = sorted(list(set(combinazioni_utente.get('estratto', []))))
        ambi_u_tpl = sorted(list(set(tuple(sorted(a)) for a in combinazioni_utente.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
        terni_u_tpl = sorted(list(set(tuple(sorted(t)) for t in combinazioni_utente.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
        quaterne_u_tpl = sorted(list(set(tuple(sorted(q)) for q in combinazioni_utente.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
        cinquine_u_tpl = sorted(list(set(tuple(sorted(c)) for c in combinazioni_utente.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))

    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver = {}; ruote_valide = []
    is_online = data_source_var.get() == "Online"
    for nome_rv_loop in nomi_ruote_verifica:
        source = URL_RUOTE.get(nome_rv_loop.upper()) if is_online else file_ruote.get(nome_rv_loop.upper())
        df_ver = carica_dati(source, start_date=start_ts, end_date=None)
        if df_ver is not None and not df_ver.empty:
            df_cache_ver[nome_rv_loop] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True); ruote_valide.append(nome_rv_loop)

    if not ruote_valide: return "Errore: Nessuna ruota di verifica valida per caricare i dati del periodo."

    sorted_date_eventi_spia = sorted(list(date_eventi_spia)); num_casi_eventi_spia_base = len(sorted_date_eventi_spia)
    if num_casi_eventi_spia_base == 0: return "Nessun caso di Evento Spia da verificare."

    out = [f"\n\n=== {titolo_sezione} ==="]
    out.append(f"Numero di Eventi Spia di base: {num_casi_eventi_spia_base}"); out.append(f"Ruote di verifica considerate: {', '.join(ruote_valide) or 'Nessuna'}")

    esiti_dettagliati_per_item = {}; items_config_da_verificare = []
    if estratti_u: items_config_da_verificare.extend([('estratto', e) for e in estratti_u])
    if ambi_u_tpl: items_config_da_verificare.extend([('ambo', a) for a in ambi_u_tpl])
    if terni_u_tpl: items_config_da_verificare.extend([('terno', t) for t in terni_u_tpl])
    if quaterne_u_tpl: items_config_da_verificare.extend([('quaterna', q) for q in quaterne_u_tpl])
    if cinquine_u_tpl: items_config_da_verificare.extend([('cinquina', c) for c in cinquine_u_tpl])

    sfaldamenti_totali_per_item_ruota = {}

    for tipo_sorte_item, item_val in items_config_da_verificare:
        item_str_key = format_ambo_terno(item_val) if isinstance(item_val, tuple) else item_val
        esiti_dettagliati_per_item[item_str_key] = []
        num_eventi_spia_coperti_per_item = 0
        total_actual_hits_for_item = 0

        sfaldamenti_totali_per_item_ruota[item_str_key] = Counter()

        for data_t in sorted_date_eventi_spia:
            dettagli_uscita_per_questo_evento_spia_lista = []
            evento_spia_coperto_in_questo_ciclo_per_item = False
            max_colpi_effettivi_per_evento_spia = 0
            ruote_con_hit_gia_contate_per_questo_evento_spia = set()

            for nome_rv in ruote_valide:
                df_v = df_cache_ver.get(nome_rv);
                if df_v is None: continue
                date_s_v = df_v['Data']
                try: start_idx = date_s_v.searchsorted(data_t, side='right')
                except Exception: continue
                if start_idx >= len(date_s_v): continue

                df_fin_v = df_v.iloc[start_idx : start_idx + n_verifiche];
                max_colpi_effettivi_per_evento_spia = max(max_colpi_effettivi_per_evento_spia, len(df_fin_v))

                if not df_fin_v.empty:
                    for colpo_idx, (_, row) in enumerate(df_fin_v.iterrows(), 1):
                        data_estrazione_corrente = row['Data'].date();
                        current_row_numbers_v = [row[col] for col in cols_num if pd.notna(row[col])];
                        nums_draw_set = set(current_row_numbers_v)
                        match = False
                        if tipo_sorte_item == 'estratto':
                            if item_val in nums_draw_set: match = True
                        elif tipo_sorte_item == 'ambo':
                            if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'terno':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'quaterna':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'cinquina':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True

                        if match:
                            total_actual_hits_for_item += 1
                            dettagli_uscita_per_questo_evento_spia_lista.append(f"{nome_rv} al colpo {colpo_idx} (data esito {data_estrazione_corrente.strftime('%d/%m')})")
                            evento_spia_coperto_in_questo_ciclo_per_item = True

                            if nome_rv not in ruote_con_hit_gia_contate_per_questo_evento_spia:
                               sfaldamenti_totali_per_item_ruota[item_str_key][nome_rv] += 1
                               ruote_con_hit_gia_contate_per_questo_evento_spia.add(nome_rv)
                            break

            esito_per_questo_evento_spia_str = ""
            if evento_spia_coperto_in_questo_ciclo_per_item:
                esito_per_questo_evento_spia_str = "; ".join(dettagli_uscita_per_questo_evento_spia_lista)
                num_eventi_spia_coperti_per_item +=1
            elif max_colpi_effettivi_per_evento_spia < n_verifiche:
                esito_per_questo_evento_spia_str = f"IN CORSO (max {max_colpi_effettivi_per_evento_spia}/{n_verifiche} colpi analizzabili)"
            else:
                esito_per_questo_evento_spia_str = "NON USCITO"

            esiti_dettagliati_per_item[item_str_key].append(f"    Evento Spia del {data_t.strftime('%d/%m/%y')}: {esito_per_questo_evento_spia_str}")

        out.append(f"\n--- Esiti {tipo_sorte_item.upper()} ---");
        out.append(f"  - {item_str_key}:");
        out.extend(esiti_dettagliati_per_item[item_str_key])

        out.append(f"\n    RIEPILOGO SFALDAMENTI SU RUOTA per {item_str_key.upper()}:")
        if not ruote_valide:
             out.append("      Nessuna ruota di verifica attiva.")
        else:
            for nome_rv_riepilogo in ruote_valide:
                conteggio = sfaldamenti_totali_per_item_ruota[item_str_key].get(nome_rv_riepilogo, 0)
                out.append(f"      - {nome_rv_riepilogo}: {conteggio} volt{'a' if conteggio == 1 else 'e'}")

        if num_casi_eventi_spia_base == 0:
            out.append(f"    RIEPILOGO GENERALE {item_str_key}: Nessun evento spia registrato per l'analisi.")
        else:
            perc_copertura_eventi_spia = (num_eventi_spia_coperti_per_item / num_casi_eventi_spia_base * 100) if num_casi_eventi_spia_base > 0 else 0
            out.append(f"    RIEPILOGO GENERALE {item_str_key}: Uscito {total_actual_hits_for_item} volte su {num_casi_eventi_spia_base} eventi spia ({perc_copertura_eventi_spia:.1f}%)")
            
    return "\n".join(out)

def verifica_esiti_futuri(top_combinati_input, nomi_ruote_verifica, data_fine_analisi, n_colpi_futuri):
    if not top_combinati_input or not any(top_combinati_input.values()) or not nomi_ruote_verifica or data_fine_analisi is None or n_colpi_futuri <= 0:
        return "Errore: Input invalidi per verifica_esiti_futuri (post-analisi)."

    is_online = data_source_var.get() == "Online"

    estratti_items = top_combinati_input.get('estratto', [])
    ambi_items = [tuple(a.split('-')) for a in top_combinati_input.get('ambo', [])]
    terni_items = [tuple(t.split('-')) for t in top_combinati_input.get('terno', [])]
    quaterne_items = [tuple(q.split('-')) for q in top_combinati_input.get('quaterna', [])]
    cinquine_items = [tuple(c.split('-')) for c in top_combinati_input.get('cinquina', [])]

    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver_fut = {}; ruote_con_dati_fut = []
    for nome_rv in nomi_ruote_verifica:
        source = URL_RUOTE.get(nome_rv.upper()) if is_online else file_ruote.get(nome_rv.upper())
        df_ver_full = carica_dati(source, start_date=None, end_date=None)
        if df_ver_full is None or df_ver_full.empty: continue
        df_ver_fut_ruota = df_ver_full[df_ver_full['Data'] > data_fine_analisi].copy().sort_values(by='Data').reset_index(drop=True)
        df_fin_fut_ruota = df_ver_fut_ruota.head(n_colpi_futuri)
        if not df_fin_fut_ruota.empty: df_cache_ver_fut[nome_rv] = df_fin_fut_ruota; ruote_con_dati_fut.append(nome_rv)

    if not ruote_con_dati_fut: return f"Nessuna estrazione trovata su nessuna ruota di verifica dopo {data_fine_analisi.strftime('%d/%m/%Y')} per {n_colpi_futuri} colpi."

    num_colpi_globalmente_analizzabili = n_colpi_futuri
    if ruote_con_dati_fut:
        min_len = min(len(df_cache_ver_fut.get(r, pd.DataFrame())) for r in ruote_con_dati_fut)
        num_colpi_globalmente_analizzabili = min_len

    hits_registrati = {
        'estratto': {e: None for e in estratti_items},
        'ambo': {a: None for a in ambi_items},
        'terno': {t: None for t in terni_items},
        'quaterna': {q: None for q in quaterne_items},
        'cinquina': {c: None for c in cinquine_items}
    }

    for nome_rv in ruote_con_dati_fut:
        df_finestra_ruota = df_cache_ver_fut[nome_rv]
        for colpo_idx, (_, row) in enumerate(df_finestra_ruota.iterrows(), 1):
            data_estrazione_corrente = row['Data'].date()
            set_numeri_riga = {str(row[col]).zfill(2) for col in cols_num if pd.notna(row[col])}

            for item_e in estratti_items:
                if hits_registrati['estratto'].get(item_e) is None and item_e in set_numeri_riga:
                    hits_registrati['estratto'][item_e] = (nome_rv, colpo_idx, data_estrazione_corrente)

            if len(set_numeri_riga) >= 2:
                for item_a in ambi_items:
                    if hits_registrati['ambo'].get(item_a) is None and set(item_a).issubset(set_numeri_riga):
                        hits_registrati['ambo'][item_a] = (nome_rv, colpo_idx, data_estrazione_corrente)

            if len(set_numeri_riga) >= 3:
                for item_t in terni_items:
                    if hits_registrati['terno'].get(item_t) is None and set(item_t).issubset(set_numeri_riga):
                        hits_registrati['terno'][item_t] = (nome_rv, colpo_idx, data_estrazione_corrente)

            if len(set_numeri_riga) >= 4:
                for item_q in quaterne_items:
                     if hits_registrati['quaterna'].get(item_q) is None and set(item_q).issubset(set_numeri_riga):
                        hits_registrati['quaterna'][item_q] = (nome_rv, colpo_idx, data_estrazione_corrente)

            if len(set_numeri_riga) >= 5:
                for item_c in cinquine_items:
                    if hits_registrati['cinquina'].get(item_c) is None and set(item_c).issubset(set_numeri_riga):
                        hits_registrati['cinquina'][item_c] = (nome_rv, colpo_idx, data_estrazione_corrente)

    out = [f"\n\n=== VERIFICA ESITI FUTURI (POST-ANALISI) ({n_colpi_futuri} Colpi dopo {data_fine_analisi.strftime('%d/%m/%Y')}) ==="]
    out.append(f"Ruote verificate con dati futuri disponibili: {', '.join(ruote_con_dati_fut) or 'Nessuna'}")
    if ruote_con_dati_fut: out.append(f"(Analisi basata su un minimo di {num_colpi_globalmente_analizzabili} colpi disponibili globalmente su queste ruote)")

    sorti_config = [
        ('estratto', estratti_items),
        ('ambo', ambi_items),
        ('terno', terni_items),
        ('quaterna', quaterne_items),
        ('cinquina', cinquine_items)
    ]

    for tipo_sorte, lista_items in sorti_config:
        if not lista_items: continue
        out.append(f"\n--- Esiti Futuri {tipo_sorte.upper()} ---")
        almeno_un_hit_per_sorte = False
        for item_da_verificare in lista_items:
            item_str_formattato = format_ambo_terno(item_da_verificare)
            dettaglio_hit = hits_registrati[tipo_sorte].get(item_da_verificare)
            if dettaglio_hit:
                almeno_un_hit_per_sorte = True
                d_ruota, d_colpo, d_data = dettaglio_hit
                out.append(f"    - {item_str_formattato}: USCITO -> {d_ruota} @ C{d_colpo} ({d_data.strftime('%d/%m/%Y')})")
            else:
                if num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut:
                    out.append(f"    - {item_str_formattato}: IN CORSO (analizzati min {num_colpi_globalmente_analizzabili}/{n_colpi_futuri} colpi)")
                else:
                    out.append(f"    - {item_str_formattato}: NON uscito")
        if not almeno_un_hit_per_sorte and lista_items and not (num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut):
            out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito nei colpi futuri analizzati.")

    return "\n".join(out)

def esegui_verifica_futura():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica_futura, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, "\n\nVerifica esiti futuri (post-analisi) in corso...");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    nomi_rv = info_ricerca_globale.get('ruote_verifica')
    data_fine = info_ricerca_globale.get('end_date')

    if not nomi_rv or not data_fine:
        messagebox.showerror("Errore Verifica Futura", "Dati di base per l'analisi (Ruote verifica, Data Fine) mancanti.");
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Futura.");risultato_text.config(state=tk.NORMAL); return

    items_da_verificare = {'estratto': [], 'ambo': [], 'terno': [], 'quaterna': [], 'cinquina': []}
    
    # === INIZIO BLOCCO CORRETTO ===
    # Aggiungiamo gli estratti alla verifica
    estratti_g = info_ricerca_globale.get('migliori_estratti_copertura_globale', [])
    if estratti_g:
        items_da_verificare['estratto'] = [item['estratto'] for item in estratti_g]

    # Aggiungiamo gli ambi alla verifica
    ambi_g = info_ricerca_globale.get('migliori_ambi_copertura_globale', [])
    if ambi_g:
        items_da_verificare['ambo'] = [item['ambo'] for item in ambi_g]
    # === FINE BLOCCO CORRETTO ===

    combinazioni_per_ambo = [
        ('migliori_terni_per_ambo_copertura_globale', 'terno'),
        ('migliori_quartine_per_ambo_copertura_globale', 'quaterna'),
        ('migliori_cinquine_per_ambo_copertura_globale', 'cinquina'),
    ]
    for key_info, key_items in combinazioni_per_ambo:
        info = info_ricerca_globale.get(key_info, [])
        if info:
            combinazioni_derivate = {format_ambo_terno(item['combinazione']) for item in info}
            items_da_verificare[key_items] = sorted(list(set(items_da_verificare.get(key_items, [])) | combinazioni_derivate))


    if not any(items_da_verificare.values()):
        messagebox.showinfo("Verifica Futura", "Nessun risultato di 'Copertura Globale' trovato dall'analisi precedente da poter verificare.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessun dato di Copertura Globale da verificare.");risultato_text.config(state=tk.NORMAL); return

    try:
        n_colpi_fut = int(estrazioni_entry_verifica_futura.get())
        assert 1 <= n_colpi_fut <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Futura (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica = ""
    try:
        res_str = verifica_esiti_futuri(items_da_verificare, nomi_rv, data_fine, n_colpi_fut)
        output_verifica = res_str
        if res_str and "Errore" not in res_str and "Nessuna estrazione trovata" not in res_str :
            mostra_popup_testo_semplice("Riepilogo Verifica Predittiva (Post-Analisi)", res_str)
    except Exception as e:
        output_verifica = f"\nErrore durante la verifica esiti futuri: {e}"
        traceback.print_exc()

    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, output_verifica)
    risultato_text.see(tk.END)

def esegui_verifica_mista():
    global info_ricerca_globale, risultato_text, root, text_combinazioni_miste, estrazioni_entry_verifica_mista, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "\n\nVerifica mista (combinazioni utente su eventi spia) in corso...");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    input_text = text_combinazioni_miste.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Input Invalido", "Nessuna combinazione inserita.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione inserita.");risultato_text.config(state=tk.NORMAL); return

    combinazioni_sets = {'estratto': set(),'ambo': set(),'terno': set(),'quaterna': set(),'cinquina': set()}; righe_input_originali = []
    for riga_idx, riga in enumerate(input_text.splitlines()):
        riga_proc = riga.strip();
        if not riga_proc: continue
        righe_input_originali.append(riga_proc)
        try:
            numeri_str = riga_proc.split('-') if '-' in riga_proc else riga_proc.split(); numeri_int = [int(n.strip()) for n in numeri_str if n.strip().isdigit()]
            if not numeri_int: raise ValueError("Nessun numero valido sulla riga.")
            if not all(1<=n<=90 for n in numeri_int): raise ValueError("Numeri fuori range (1-90).")
            if len(set(numeri_int))!=len(numeri_int): raise ValueError("Numeri duplicati.")
            if not (1<=len(numeri_int)<=5): raise ValueError("Inserire da 1 a 5 numeri.")

            numeri_validi_zfill = sorted([str(n).zfill(2) for n in numeri_int]); num_elementi = len(numeri_validi_zfill)

            if num_elementi == 1:
                combinazioni_sets['estratto'].add(numeri_validi_zfill[0])
            elif num_elementi == 2:
                combinazioni_sets['ambo'].add(tuple(numeri_validi_zfill))
            elif num_elementi == 3:
                combinazioni_sets['terno'].add(tuple(numeri_validi_zfill))
                for ambo_comb in itertools.combinations(numeri_validi_zfill, 2):
                    combinazioni_sets['ambo'].add(tuple(sorted(ambo_comb)))
            elif num_elementi == 4:
                combinazioni_sets['quaterna'].add(tuple(numeri_validi_zfill))
                for terno_comb in itertools.combinations(numeri_validi_zfill, 3):
                    combinazioni_sets['terno'].add(tuple(sorted(terno_comb)))
                for ambo_comb in itertools.combinations(numeri_validi_zfill, 2):
                    combinazioni_sets['ambo'].add(tuple(sorted(ambo_comb)))
            elif num_elementi == 5:
                combinazioni_sets['cinquina'].add(tuple(numeri_validi_zfill))
                for quaterna_comb in itertools.combinations(numeri_validi_zfill, 4):
                    combinazioni_sets['quaterna'].add(tuple(sorted(quaterna_comb)))
                for terno_comb in itertools.combinations(numeri_validi_zfill, 3):
                    combinazioni_sets['terno'].add(tuple(sorted(terno_comb)))
                for ambo_comb in itertools.combinations(numeri_validi_zfill, 2):
                    combinazioni_sets['ambo'].add(tuple(sorted(ambo_comb)))

        except ValueError as ve:
            messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {ve}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return
        except Exception as e_parse:
            messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {e_parse}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return

    combinazioni_utente = {k: sorted(list(v)) for k, v in combinazioni_sets.items() if v}
    if not any(combinazioni_utente.values()):
        messagebox.showerror("Input Invalido","Nessuna combinazione valida estratta.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione valida.");risultato_text.config(state=tk.NORMAL);return

    date_eventi_spia = info_ricerca_globale.get('date_eventi_spia_ordinate')
    nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date')
    end_ts = info_ricerca_globale.get('end_date')
    numeri_spia_originali = info_ricerca_globale.get('numeri_spia_input', [])
    spia_display_originale = format_ambo_terno(numeri_spia_originali) if isinstance(numeri_spia_originali, (list,tuple)) else str(numeri_spia_originali)

    if not all([date_eventi_spia, nomi_rv, start_ts is not None, end_ts is not None]):
        messagebox.showerror("Errore Verifica Mista", "Dati analisi 'Successivi' mancanti. Eseguire prima un'analisi.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Mista (dati mancanti).");risultato_text.config(state=tk.NORMAL); return

    try:
        n_colpi_misti = int(estrazioni_entry_verifica_mista.get())
        assert 1 <= n_colpi_misti <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Mista (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica_mista = ""
    try:
        titolo_output = f"VERIFICA MISTA (COMBINAZIONI UTENTE) - Dopo Spia: {spia_display_originale} ({n_colpi_misti} Colpi dopo ogni Evento Spia)"
        res_str = verifica_esiti_utente_su_eventi_spia(date_eventi_spia, combinazioni_utente, nomi_rv, n_colpi_misti, start_ts, end_ts, titolo_sezione=titolo_output)

        summary_input = "\nNumeri scelti per Verifica Mista:\n" + "\n".join([f"  - {r}" for r in righe_input_originali])

        lines = res_str.splitlines(); insert_idx_summary = 1
        final_output_lines = lines[:insert_idx_summary] + [summary_input] + lines[insert_idx_summary:]
        output_verifica_mista = "\n".join(final_output_lines)

        if output_verifica_mista and "Errore" not in output_verifica_mista and "Nessun caso evento spia" not in output_verifica_mista:
            mostra_popup_testo_semplice(f"Riepilogo Verifica Mista (Spia: {spia_display_originale})", output_verifica_mista)
    except Exception as e:
        output_verifica_mista = f"\nErrore durante la verifica mista: {e}"
        traceback.print_exc()

    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, output_verifica_mista)
    risultato_text.see(tk.END)
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and bool(risultati_globali) and any(r[2] for r in risultati_globali if len(r)>2):
        valid_res = [r for r in risultati_globali if r[2] is not None]
        if valid_res: visualizza_grafici(valid_res, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni',5))
        else: messagebox.showinfo("Grafici", "Nessun risultato valido per grafici.")
    else: messagebox.showinfo("Grafici", "Esegui 'Cerca Successivi' con risultati validi prima.")

def trova_abbinamenti_numero_target(numero_target_str, nomi_ruote_ricerca, start_ts, end_ts, top_n_simpatici):
    global URL_RUOTE, file_ruote, data_source_var
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    if not numero_target_str or not (numero_target_str.isdigit() and 1 <= int(numero_target_str) <= 90): return None, "Numero target non valido (deve essere 1-90).", 0
    numero_target_zfill = numero_target_str.zfill(2); abbinamenti_counter = Counter(); occorrenze_target = 0; ruote_effettivamente_analizzate = []
    is_online = data_source_var.get() == "Online"

    for nome_ruota in nomi_ruote_ricerca:
        source = URL_RUOTE.get(nome_ruota.upper()) if is_online else file_ruote.get(nome_ruota.upper())
        if not source:
            print(f"Attenzione: Fonte dati per ruota {nome_ruota} non trovata. Saltata.")
            continue
        df_ruota = carica_dati(source, start_date=start_ts, end_date=end_ts)
        if df_ruota is None or df_ruota.empty: continue
        ruote_effettivamente_analizzate.append(nome_ruota)
        for _, row in df_ruota.iterrows():
            numeri_estratti_riga = sorted([row[col] for col in colonne_numeri if pd.notna(row[col])])
            if numero_target_zfill in numeri_estratti_riga:
                occorrenze_target += 1
                altri_numeri_nella_stessa_estrazione = [n for n in numeri_estratti_riga if n != numero_target_zfill]
                if altri_numeri_nella_stessa_estrazione: abbinamenti_counter.update(altri_numeri_nella_stessa_estrazione)
    if not ruote_effettivamente_analizzate: return None, "Nessuna ruota selezionata conteneva dati validi.", 0
    if occorrenze_target == 0: return [], f"Numero target '{numero_target_zfill}' non trovato.", 0
    return abbinamenti_counter.most_common(top_n_simpatici), None, occorrenze_target

def esegui_ricerca_numeri_simpatici():
    global risultato_text, root, entry_numero_target_simpatici, listbox_ruote_simpatici
    global entry_top_n_simpatici, start_date_entry, end_date_entry, data_source_var

    if data_source_var.get() == "Locale" and (not mappa_file_ruote() or not file_ruote):
        messagebox.showerror("Errore Cartella", "Modalità 'Locale' selezionata, ma la cartella non è valida.")
        return

    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "Ricerca Numeri Simpatici in corso...\n");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    try:
        numero_target = entry_numero_target_simpatici.get().strip()
        if not numero_target or not numero_target.isdigit() or not (1 <= int(numero_target) <= 90): messagebox.showerror("Input Invalido", "Numero Scelto deve essere tra 1 e 90.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Numero Scelto non valido.");risultato_text.config(state=tk.NORMAL);return
        selected_ruote_indices = listbox_ruote_simpatici.curselection(); nomi_ruote_selezionate_final = []
        all_ruote_in_listbox = [listbox_ruote_simpatici.get(i) for i in range(listbox_ruote_simpatici.size())]; valid_ruote_from_listbox = [r for r in all_ruote_in_listbox if r not in ["Nessuna ruota configurata", "Nessun file ruota valido"]]
        if not selected_ruote_indices:
            if not valid_ruote_from_listbox: messagebox.showerror("Input Invalido", "Nessuna ruota valida disponibile.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Ruote non valide.");risultato_text.config(state=tk.NORMAL);return
            nomi_ruote_selezionate_final = valid_ruote_from_listbox
        else: nomi_ruote_selezionate_final = [listbox_ruote_simpatici.get(i) for i in selected_ruote_indices]
        top_n_str = entry_top_n_simpatici.get().strip()
        if not top_n_str.isdigit() or int(top_n_str) <= 0: messagebox.showerror("Input Invalido", "Il numero di 'Top N' deve essere un intero positivo.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Top N non valido.");risultato_text.config(state=tk.NORMAL);return
        top_n = int(top_n_str)
        start_dt = start_date_entry.get_date(); end_dt = end_date_entry.get_date()
        if start_dt > end_dt: messagebox.showerror("Input Date", "Data di inizio non può essere successiva alla data di fine.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Date non valide.");risultato_text.config(state=tk.NORMAL);return
        start_ts = pd.Timestamp(start_dt); end_ts = pd.Timestamp(end_dt)
    except Exception as e: messagebox.showerror("Errore Input", f"Errore input: {e}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input: {e}");risultato_text.config(state=tk.NORMAL);return

    risultati_simpatici, errore_msg, occorrenze_target = trova_abbinamenti_numero_target(numero_target, nomi_ruote_selezionate_final, start_ts, end_ts, top_n)

    output_lines = [f"=== Risultati Ricerca Numeri Simpatici ===", f"Numero Scelto Analizzato: {numero_target.zfill(2)}", f"Ruote Analizzate: {', '.join(nomi_ruote_selezionate_final) if nomi_ruote_selezionate_final else 'Nessuna ruota specificata o valida'}", f"Periodo: {start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}"]
    if errore_msg: output_lines.append(f"\nMessaggio dal sistema: {errore_msg}")
    if risultati_simpatici is None: output_lines.append("Ricerca fallita o interrotta.")
    elif occorrenze_target == 0:
        if not errore_msg: output_lines.append(f"Il Numero Scelto '{numero_target.zfill(2)}' non è stato trovato.")
    else:
        output_lines.append(f"Il Numero Scelto '{numero_target.zfill(2)}' è stato trovato {occorrenze_target} volte.")
        if not risultati_simpatici: output_lines.append("Nessun altro numero abbinato trovato.")
        else:
            output_lines.append(f"\nTop {len(risultati_simpatici)} Numeri Simpatici (Numero: Frequenza Abbinamento):")
            for i, (num_simp, freq) in enumerate(risultati_simpatici): output_lines.append(f"  {i+1}. Numero {str(num_simp).zfill(2)}: {freq} volte")

    final_output_str = "\n".join(output_lines)
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, final_output_str)
    risultato_text.see(tk.END)

    if not errore_msg and risultati_simpatici is not None and occorrenze_target > 0:
         mostra_popup_testo_semplice(f"Numeri Simpatici per {numero_target.zfill(2)}", final_output_str)
    elif errore_msg and (risultati_simpatici is not None or occorrenze_target == 0):
        mostra_popup_testo_semplice(f"Info Numeri Simpatici per {numero_target.zfill(2)}", final_output_str)

# =============================================================================
# GUI e Mainloop
# =============================================================================
root = tk.Tk()
root.title("Numeri Spia - Marker - Simpatici - Created by Massimo Ferrughelli- IL LOTTO DI MAX - ")
root.geometry("1350x850")
root.minsize(1200, 750)
root.configure(bg="#f0f0f0")

style = ttk.Style(); style.theme_use('clam')
style.configure("TFrame", background="#f0f0f0"); style.configure("TLabel", background="#f0f0f0", font=("Segoe UI",10))
style.configure("TButton", font=("Segoe UI",10), padding=5); style.configure("Title.TLabel", font=("Segoe UI",11,"bold"))
style.configure("Header.TLabel", font=("Segoe UI",12,"bold")); style.configure("Small.TLabel", background="#f0f0f0", font=("Segoe UI",8))
style.configure("TEntry", padding=3); style.configure("TListbox", font=("Consolas",10)); style.configure("TLabelframe.Label", font=("Segoe UI",10,"bold"), background="#f0f0f0")
style.configure("TNotebook.Tab", padding=[10,5], font=("Segoe UI",10))
style.configure("TRadiobutton", background="#f0f0f0", font=("Segoe UI", 9))


# =============================================================================
# === INIZIO BLOCCO MODIFICATO: Rimozione Canvas e Scrollbar Principale ===
# =============================================================================
# Il meccanismo del canvas e della scrollbar principale è stato rimosso.
# Ora il 'main_frame' è direttamente figlio della finestra 'root'.
# Questo impedisce la comparsa della scrollbar verticale principale.
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)
# =============================================================================
# === FINE BLOCCO MODIFICATO ===
# =============================================================================


# --- Controlli per la selezione della fonte dati ---
source_frame_outer = ttk.LabelFrame(main_frame, text=" Fonte Dati ", padding=10)
source_frame_outer.pack(fill=tk.X, pady=(0, 10))
data_source_var = tk.StringVar(value="Online")

cartella_frame = ttk.Frame(source_frame_outer)
ttk.Label(cartella_frame, text="Cartella Locale:", style="Title.TLabel").pack(side=tk.LEFT, padx=(0,5))
cartella_entry = ttk.Entry(cartella_frame, width=60)
cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
btn_sfoglia = ttk.Button(cartella_frame, text="Sfoglia...")
btn_sfoglia.pack(side=tk.LEFT, padx=5)

def toggle_data_source_ui():
    is_local = (data_source_var.get() == "Locale")

    if is_local:
        for widget in cartella_frame.winfo_children():
            widget.config(state=tk.NORMAL)
    else:
        for widget in cartella_frame.winfo_children():
            widget.config(state=tk.DISABLED)

    if is_local:
        if not mappa_file_ruote():
            messagebox.showwarning("Cartella non valida", "Il percorso predefinito o selezionato non contiene file ruota validi.")

    listboxes_to_update = [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici]
    for lb in listboxes_to_update:
        if lb:
            aggiorna_lista_file_gui(lb)

radio_online = ttk.Radiobutton(source_frame_outer, text="Online (GitHub)", variable=data_source_var, value="Online", command=toggle_data_source_ui)
radio_online.pack(anchor='w', side=tk.LEFT, padx=5)
radio_locale = ttk.Radiobutton(source_frame_outer, text="Cartella Locale", variable=data_source_var, value="Locale", command=toggle_data_source_ui)
radio_locale.pack(anchor='w', side=tk.LEFT, padx=5)
cartella_frame.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=20)

notebook = ttk.Notebook(main_frame, style="TNotebook"); notebook.pack(fill=tk.X, pady=10)

tab_successivi = ttk.Frame(notebook, padding=10)
notebook.add(tab_successivi, text=' Analisi Numeri Successivi (Spia) ')
controls_frame_succ = ttk.Frame(tab_successivi)
controls_frame_succ.pack(fill=tk.X)
controls_frame_succ.columnconfigure(0, weight=1); controls_frame_succ.columnconfigure(1, weight=1); controls_frame_succ.columnconfigure(2, weight=0); controls_frame_succ.columnconfigure(3, weight=0)
ruote_analisi_outer_frame = ttk.Frame(controls_frame_succ); ruote_analisi_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi (Spia):", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame); ruote_analisi_list_frame.pack(fill=tk.BOTH, expand=True)
scrollbar_ra = ttk.Scrollbar(ruote_analisi_list_frame); scrollbar_ra.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_ra.set)
listbox_ruote_analisi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_ra.config(command=listbox_ruote_analisi.yview)
ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ); ruote_verifica_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5)
ttk.Label(ruote_verifica_outer_frame, text="4. Ruote Verifica (Esiti):", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame); ruote_verifica_list_frame.pack(fill=tk.BOTH, expand=True)
scrollbar_rv = ttk.Scrollbar(ruote_verifica_list_frame); scrollbar_rv.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_rv.set)
listbox_ruote_verifica.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_rv.config(command=listbox_ruote_verifica.yview)
center_controls_frame_succ = ttk.Frame(controls_frame_succ); center_controls_frame_succ.grid(row=0, column=2, sticky="ns", padx=5)
tipo_spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Tipo di Spia ", padding=5); tipo_spia_frame_succ.pack(fill=tk.X, pady=(0,5))
if 'tipo_spia_var_global' not in globals() or globals()['tipo_spia_var_global'] is None: tipo_spia_var_global = tk.StringVar(value="estratto")
def _toggle_posizione_spia_input_tab_succ():
    global tipo_spia_var_global, combo_posizione_spia
    is_posizionale = (tipo_spia_var_global.get() == "estratto_posizionale")
    if combo_posizione_spia: combo_posizione_spia.config(state="readonly" if is_posizionale else tk.DISABLED);_=[combo_posizione_spia.current(0) if is_posizionale and (not combo_posizione_spia.get() or combo_posizione_spia.get() not in combo_posizione_spia['values']) else None, combo_posizione_spia.set("Qualsiasi") if not is_posizionale else None]
ttk.Radiobutton(tipo_spia_frame_succ, text="Estratto Spia", variable=tipo_spia_var_global, value="estratto", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
ttk.Radiobutton(tipo_spia_frame_succ, text="Ambo Spia", variable=tipo_spia_var_global, value="ambo", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
ttk.Radiobutton(tipo_spia_frame_succ, text="Estratto Spia Posizionale", variable=tipo_spia_var_global, value="estratto_posizionale", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 3. Numeri Spia / Posizione ",padding=5); spia_frame_succ.pack(fill=tk.X,pady=(0,5))
spia_entry_container_succ = ttk.Frame(spia_frame_succ); spia_entry_container_succ.pack(fill=tk.X,pady=5)
entry_numeri_spia.clear(); _=[entry_numeri_spia.append(ttk.Entry(spia_entry_container_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10))) or entry_numeri_spia[-1].pack(side=tk.LEFT,padx=3,ipady=2) for _ in range(5)]
posizioni_spia_options = ["Qualsiasi", "1a", "2a", "3a", "4a", "5a"]
if 'combo_posizione_spia' not in globals() or globals()['combo_posizione_spia'] is None: combo_posizione_spia = ttk.Combobox(spia_entry_container_succ, values=posizioni_spia_options, width=8, font=("Segoe UI", 9), state="disabled"); combo_posizione_spia.set("Qualsiasi"); combo_posizione_spia.pack(side=tk.LEFT, padx=(5,0), ipady=1)
estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Estrazioni Successive ",padding=5); estrazioni_frame_succ.pack(fill=tk.X,pady=5)
ttk.Label(estrazioni_frame_succ, text=f"Quante (1-{MAX_COLPI_GIOCO}):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_succ.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_succ.insert(0,"5"); _toggle_posizione_spia_input_tab_succ()
buttons_frame_succ = ttk.Frame(controls_frame_succ); buttons_frame_succ.grid(row=0, column=3, sticky="ns", padx=(10,0))
button_cerca_succ = ttk.Button(buttons_frame_succ, text="Cerca Successivi", command=lambda:cerca_numeri(modalita="successivi")); button_cerca_succ.pack(pady=5, fill=tk.X, ipady=3)
tab_antecedenti = ttk.Frame(notebook,padding=10); notebook.add(tab_antecedenti, text=' Analisi Numeri Antecedenti (Marker) ')
controls_frame_ant = ttk.Frame(tab_antecedenti); controls_frame_ant.pack(fill=tk.X); controls_frame_ant.columnconfigure(0,weight=1); controls_frame_ant.columnconfigure(1,weight=0); controls_frame_ant.columnconfigure(2,weight=0)
ruote_analisi_ant_outer_frame = ttk.Frame(controls_frame_ant); ruote_analisi_ant_outer_frame.grid(row=0,column=0,sticky="nsew",padx=(0,10)); ttk.Label(ruote_analisi_ant_outer_frame, text="1. Ruote da Analizzare:",style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_ant_outer_frame, text="(Obiettivo e antecedenti cercati qui)",style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_ant_list_frame = ttk.Frame(ruote_analisi_ant_outer_frame); ruote_analisi_ant_list_frame.pack(fill=tk.BOTH,expand=True); scrollbar_ra_ant = ttk.Scrollbar(ruote_analisi_ant_list_frame); scrollbar_ra_ant.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_analisi_ant = tk.Listbox(ruote_analisi_ant_list_frame,height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_ra_ant.set); listbox_ruote_analisi_ant.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_ra_ant.config(command=listbox_ruote_analisi_ant.yview)
center_controls_frame_ant = ttk.Frame(controls_frame_ant); center_controls_frame_ant.grid(row=0,column=1,sticky="ns",padx=10); obiettivo_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 2. Numeri Obiettivo (1-90) ",padding=5); obiettivo_frame_ant.pack(fill=tk.X,pady=(0,5))
obiettivo_entry_container_ant = ttk.Frame(obiettivo_frame_ant); obiettivo_entry_container_ant.pack(fill=tk.X,pady=5)
if 'entry_numeri_obiettivo' not in globals() or not isinstance(globals()['entry_numeri_obiettivo'], list): entry_numeri_obiettivo = []
else: entry_numeri_obiettivo.clear()
_=[entry_numeri_obiettivo.append(ttk.Entry(obiettivo_entry_container_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10))) or entry_numeri_obiettivo[-1].pack(side=tk.LEFT,padx=3,ipady=2) for _ in range(5)]
estrazioni_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 3. Estrazioni Precedenti ",padding=5); estrazioni_frame_ant.pack(fill=tk.X,pady=5); ttk.Label(estrazioni_frame_ant, text="Quante controllare (>=1):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_ant = ttk.Entry(estrazioni_frame_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_ant.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_ant.insert(0,"3")
buttons_frame_ant = ttk.Frame(controls_frame_ant); buttons_frame_ant.grid(row=0,column=2,sticky="ns",padx=(10,0)); button_cerca_ant = ttk.Button(buttons_frame_ant, text="Cerca Antecedenti",command=lambda:cerca_numeri(modalita="antecedenti")); button_cerca_ant.pack(pady=5,fill=tk.X,ipady=3)
tab_simpatici = ttk.Frame(notebook, padding=10); notebook.add(tab_simpatici, text=' Numeri Simpatici ')
controls_frame_simpatici_outer = ttk.Frame(tab_simpatici); controls_frame_simpatici_outer.pack(fill=tk.X, pady=5); controls_frame_simpatici_outer.columnconfigure(0, weight=0); controls_frame_simpatici_outer.columnconfigure(1, weight=1)
input_params_simpatici_frame = ttk.Frame(controls_frame_simpatici_outer); input_params_simpatici_frame.grid(row=0, column=0, sticky="ns", padx=(0,10));
lbl_numero_target = ttk.Label(input_params_simpatici_frame, text="Numero Scelto (1-90):", style="Title.TLabel");
lbl_numero_target.pack(anchor="w", pady=(0,2))
entry_numero_target_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_numero_target_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_numero_target_simpatici.insert(0, "10");
lbl_top_n_simpatici = ttk.Label(input_params_simpatici_frame, text="Quanti Numeri Simpatici (Top N):", style="Title.TLabel");
lbl_top_n_simpatici.pack(anchor="w", pady=(5,2))
entry_top_n_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_top_n_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_top_n_simpatici.insert(0, "10");
button_cerca_simpatici = ttk.Button(input_params_simpatici_frame, text="Cerca Numeri Simpatici", command=esegui_ricerca_numeri_simpatici);
button_cerca_simpatici.pack(pady=10, fill=tk.X, ipady=3)
ruote_simpatici_frame_outer = ttk.Frame(controls_frame_simpatici_outer); ruote_simpatici_frame_outer.grid(row=0, column=1, sticky="nsew", padx=(10,0)); ttk.Label(ruote_simpatici_frame_outer, text="Ruote di Ricerca (CTRL/SHIFT per multiple, nessuna = tutte):", style="Title.TLabel").pack(anchor="w")
ruote_simpatici_list_frame_inner = ttk.Frame(ruote_simpatici_frame_outer); ruote_simpatici_list_frame_inner.pack(fill=tk.BOTH, expand=True, pady=(5,0)); scrollbar_rs_y = ttk.Scrollbar(ruote_simpatici_list_frame_inner, orient=tk.VERTICAL); scrollbar_rs_y.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_simpatici = tk.Listbox(ruote_simpatici_list_frame_inner, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_rs_y.set); listbox_ruote_simpatici.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_rs_y.config(command=listbox_ruote_simpatici.yview)
common_controls_top_frame = ttk.Frame(main_frame); common_controls_top_frame.pack(fill=tk.X,pady=5)
dates_frame = ttk.LabelFrame(common_controls_top_frame, text=" Periodo Analisi (Comune) ",padding=5); dates_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y); dates_frame.columnconfigure(1,weight=1)
ttk.Label(dates_frame,text="Da:",anchor="e").grid(row=0,column=0,padx=2,pady=2,sticky="w"); start_date_default = datetime.date.today()-datetime.timedelta(days=365*3)
start_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9),year=start_date_default.year,month=start_date_default.month,day=start_date_default.day); start_date_entry.grid(row=0,column=1,padx=2,pady=2,sticky="ew")
ttk.Label(dates_frame,text="A:",anchor="e").grid(row=1,column=0,padx=2,pady=2,sticky="w"); end_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9)); end_date_entry.grid(row=1,column=1,padx=2,pady=2,sticky="ew")
common_buttons_frame = ttk.Frame(common_controls_top_frame); common_buttons_frame.pack(side=tk.LEFT,padx=10,fill=tk.Y)
button_salva = ttk.Button(common_buttons_frame,text="Salva Risultati",command=salva_risultati); button_salva.pack(side=tk.LEFT,pady=5,padx=5,ipady=3)
button_visualizza = ttk.Button(common_buttons_frame,text="Visualizza Grafici\n(Solo Successivi)",command=visualizza_grafici_successivi); button_visualizza.pack(side=tk.LEFT,pady=5,padx=5,ipady=0); button_visualizza.config(state=tk.DISABLED)

post_analysis_outer_frame = ttk.Frame(main_frame)
post_analysis_outer_frame.pack(fill=tk.X, pady=(5, 0))
post_analysis_outer_frame.columnconfigure(0, weight=1)
post_analysis_outer_frame.columnconfigure(1, weight=0)
post_analysis_outer_frame.columnconfigure(2, weight=1)

# Frame per Verifica Futura (sinistra)
verifica_futura_frame = ttk.LabelFrame(post_analysis_outer_frame, text=" Verifica Predittiva (Post-Analisi) ",padding=5)
verifica_futura_frame.grid(row=0, column=0, sticky="ns", padx=(0, 5))
ttk.Label(verifica_futura_frame,text=f"Controlla N Colpi (1-{MAX_COLPI_GIOCO}):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_futura = ttk.Entry(verifica_futura_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10))
estrazioni_entry_verifica_futura.pack(anchor="w",pady=2,ipady=2)
estrazioni_entry_verifica_futura.insert(0,"9")
button_verifica_futura = ttk.Button(verifica_futura_frame,text="Verifica Futura\n(Post-Analisi)",command=esegui_verifica_futura)
button_verifica_futura.pack(pady=5,fill=tk.X,ipady=0)
button_verifica_futura.config(state=tk.DISABLED)

# Frame per Opzioni Risultati (centro) - ORDINE CORRETTO E PULITO
opzioni_risultati_frame = ttk.LabelFrame(post_analysis_outer_frame, text=" Opzioni Risultati Copertura ", padding=5)
opzioni_risultati_frame.grid(row=0, column=1, sticky="ns", padx=5)

# Riga 0: Estratti
ttk.Label(opzioni_risultati_frame, text="N. Estratti da mostrare:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
entry_num_estratti_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_estratti_popup.grid(row=0, column=1, sticky="w", padx=5, pady=2)
entry_num_estratti_popup.insert(0, "3")

# Riga 1: Ambi
ttk.Label(opzioni_risultati_frame, text="N. Ambi da mostrare:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
entry_num_ambi_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_ambi_popup.grid(row=1, column=1, sticky="w", padx=5, pady=2)
entry_num_ambi_popup.insert(0, "5")

# Riga 2: Terni
ttk.Label(opzioni_risultati_frame, text="N. Terni da mostrare:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
entry_num_terni_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_terni_popup.grid(row=2, column=1, sticky="w", padx=5, pady=2)
entry_num_terni_popup.insert(0, "5")

# Riga 3: Terni per Ambo
ttk.Label(opzioni_risultati_frame, text="N. Terni per Ambo:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
entry_num_terni_per_ambo_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_terni_per_ambo_popup.grid(row=3, column=1, sticky="w", padx=5, pady=2)
entry_num_terni_per_ambo_popup.insert(0, "3")

# Riga 4: Quartine per Ambo (AGGIUNTO)
ttk.Label(opzioni_risultati_frame, text="N. Quartine per Ambo:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
entry_num_quartine_per_ambo_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_quartine_per_ambo_popup.grid(row=4, column=1, sticky="w", padx=5, pady=2)
entry_num_quartine_per_ambo_popup.insert(0, "3")

# Riga 5: Cinquine per Ambo (AGGIUNTO)
ttk.Label(opzioni_risultati_frame, text="N. Cinquine per Ambo:").grid(row=5, column=0, sticky="w", padx=5, pady=2)
entry_num_cinquine_per_ambo_popup = ttk.Entry(opzioni_risultati_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
entry_num_cinquine_per_ambo_popup.grid(row=5, column=1, sticky="w", padx=5, pady=2)
entry_num_cinquine_per_ambo_popup.insert(0, "3")

# Riga 6 e 7: Checkbox
calcola_coperture_var = tk.BooleanVar(value=True)
check_coperture = ttk.Checkbutton(opzioni_risultati_frame, text="Calcola Coperture Globali", variable=calcola_coperture_var)
check_coperture.grid(row=6, column=0, columnspan=2, sticky='w', pady=(5,0))

calcola_ritardi_var = tk.BooleanVar(value=True)
check_ritardi = ttk.Checkbutton(opzioni_risultati_frame, text="Calcola Ritardi Attuali (Globali)", variable=calcola_ritardi_var)
check_ritardi.grid(row=7, column=0, columnspan=2, sticky='w')

# Frame per Verifica Mista (destra)
verifica_mista_frame = ttk.LabelFrame(post_analysis_outer_frame, text=" Verifica Mista (su Eventi Spia) ",padding=5)
verifica_mista_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
ttk.Label(verifica_mista_frame, text="Combinazioni (1-5 numeri, 1-90, una per riga):", style="Small.TLabel").pack(anchor="w")
text_mista_container = ttk.Frame(verifica_mista_frame); text_mista_container.pack(fill=tk.BOTH, expand=True, pady=(0,5)); text_combinazioni_miste_scrollbar_y = ttk.Scrollbar(text_mista_container, orient=tk.VERTICAL); text_combinazioni_miste_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_combinazioni_miste = tk.Text(text_mista_container, height=3, width=30, font=("Consolas", 10), wrap=tk.WORD, yscrollcommand=text_combinazioni_miste_scrollbar_y.set, bd=1, relief="sunken"); text_combinazioni_miste.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); text_combinazioni_miste_scrollbar_y.config(command=text_combinazioni_miste.yview); text_combinazioni_miste.insert("1.0", "18 23 43\n60")
ttk.Label(verifica_mista_frame, text=f"Controlla N Colpi Verifica (1-{MAX_COLPI_GIOCO}):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_mista = ttk.Entry(verifica_mista_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_mista.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_mista.insert(0,"9")
button_verifica_mista = ttk.Button(verifica_mista_frame,text="Verifica Mista\n(su Eventi Spia)",command=esegui_verifica_mista); button_verifica_mista.pack(pady=5,fill=tk.X,ipady=0); button_verifica_mista.config(state=tk.DISABLED)

ttk.Label(main_frame, text="Risultati Analisi (Log):", style="Header.TLabel").pack(anchor="w",pady=(15,0))
risultato_outer_frame = ttk.Frame(main_frame)
risultato_outer_frame.pack(fill=tk.BOTH,expand=True,pady=5)

risultato_text = tk.Text(
    risultato_outer_frame,
    wrap=tk.WORD,
    font=("Consolas", 10),
    height=15,
    state=tk.NORMAL,
    bd=1,
    relief="sunken"
)
risultato_text.pack(fill=tk.BOTH,expand=True)


def mappa_file_ruote():
    global file_ruote, cartella_entry
    cartella = cartella_entry.get(); file_ruote.clear()
    if not cartella or not os.path.isdir(cartella): return False
    ruote_valide = list(RUOTE_NOMI_MAPPATURA.keys()); found = False
    try:
        for file in os.listdir(cartella):
            fp = os.path.join(cartella, file)
            if os.path.isfile(fp) and file.lower().endswith(".txt"):
                nome_base = os.path.splitext(file)[0].upper()
                if nome_base in ruote_valide:
                    file_ruote[nome_base] = fp
                    found = True
        return found
    except OSError as e: print(f"Errore lettura cartella: {e}"); return False
    except Exception as e: print(f"Errore scansione file: {e}"); traceback.print_exc(); return False

def aggiorna_lista_file_gui(target_listbox):
    global URL_RUOTE, file_ruote, data_source_var
    if not target_listbox: return
    target_listbox.config(state=tk.NORMAL); target_listbox.delete(0, tk.END)

    if data_source_var.get() == "Online":
        lista_ruote_originale = list(URL_RUOTE.keys())
    else: # Locale
        lista_ruote_originale = list(file_ruote.keys())

    ruota_nazionale_str = "NAZIONALE"
    ruote_ordinate = sorted([r for r in lista_ruote_originale if r != ruota_nazionale_str])

    if ruota_nazionale_str in lista_ruote_originale:
        ruote_ordinate.append(ruota_nazionale_str)

    if ruote_ordinate:
        for r in ruote_ordinate:
            target_listbox.insert(tk.END, r)
    else:
        msg = "Nessuna ruota configurata" if data_source_var.get() == "Online" else "Nessun file ruota valido"
        target_listbox.insert(tk.END, msg)
        target_listbox.config(state=tk.DISABLED)

def on_sfoglia_click():
    global cartella_entry
    cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni")
    if cartella_sel:
        cartella_entry.delete(0,tk.END); cartella_entry.insert(0,cartella_sel)
        if mappa_file_ruote():
            listboxes = [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici]
            for lb in listboxes:
                if lb:
                    aggiorna_lista_file_gui(lb)
        else:
            messagebox.showwarning("Nessun File", "Nessun file .txt valido trovato nella cartella selezionata.")

def main():
    global root, risultato_text, MAX_COLPI_GIOCO, cartella_entry, btn_sfoglia
    global calcola_ritardo_globale_var
    global entry_num_estratti_popup, entry_num_ambi_popup, entry_num_terni_popup, entry_num_terni_per_ambo_popup
    global entry_num_quartine_per_ambo_popup, entry_num_cinquine_per_ambo_popup

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    default_folder_path = os.path.join(desktop_path, "NUMERICAL_EMPATHY_COMPLETO_2025")
    cartella_entry.insert(0, default_folder_path)
    btn_sfoglia.config(command=on_sfoglia_click)

    calcola_ritardo_globale_var = tk.BooleanVar(value=True)

    # --- Riorganizzazione GUI per i nuovi campi ---
    # Questa sezione è già stata creata sopra, la ridefinizione qui potrebbe causare problemi.
    # Assicuriamoci che 'entry_num_estratti_popup' e gli altri siano definiti una sola volta.
    # L'unico campo mancante nel layout originale era entry_num_estratti_popup, lo aggiungiamo.
    opzioni_frame = post_analysis_outer_frame.grid_slaves(row=0, column=1)[0] # Otteniamo il frame già creato
    
    ttk.Label(opzioni_frame, text="N. Estratti da mostrare:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    entry_num_estratti_popup = ttk.Entry(opzioni_frame, width=5, justify=tk.CENTER, font=("Segoe UI",10))
    entry_num_estratti_popup.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    entry_num_estratti_popup.insert(0, "3")
    

    toggle_data_source_ui()

    listboxes = [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici]
    for lb in listboxes:
        if lb:
            aggiorna_lista_file_gui(lb)

    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    welcome_message = (
        f"Ciao Max...cosa studiamo oggi?\n\n"
        "1. Scegli la 'Fonte Dati': Online (da GitHub) o Locale (dal tuo PC).\n"
        "   - Se 'Locale', assicurati che il percorso della cartella sia corretto.\n"
        "2. Imposta il periodo di analisi.\n"
        "3. Scegli la modalità di analisi e clicca il relativo pulsante 'Cerca...'\n\n"
    )
    risultato_text.insert(tk.END, welcome_message)
    root.mainloop()
    print("\nScript terminato.")

if __name__ == "__main__":
    main()