# -*- coding: utf-8 -*-
# Versione 3.9.5 - AGGIUNTA POPUP PER VERIFICA PREDITTIVA E MISTA (CODICE COMPLETO)
# + RIMOZIONE VERIFICA ESITI CLASSICA
# + CORREZIONE IMPORT SCROLLEDTEXT
# + CORREZIONE AVANZATA ERRORE SORT_INDEX PANDAS
# + AGGIUNTA POPUP RISULTATI ANALISI SPIA
# + modifiche GUI per selezionare tipo di spia

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkinter import scrolledtext # IMPORT CORRETTO
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

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}

# Variabili GUI globali
# button_verifica_esiti = None # RIMOSSO
button_visualizza = None
button_verifica_futura = None
estrazioni_entry_verifica_futura = None
button_verifica_mista = None
text_combinazioni_miste = None
estrazioni_entry_verifica_mista = None
tipo_spia_var_global = None
estrazioni_entry_verifica = None # Per il frame che era della verifica classica (ora disabilitato)


# =============================================================================
# FUNZIONI GRAFICHE (Come da tua versione 3.9.3)
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
            base_conteggio = risultato.get('totale_trigger', 0)
            titolo = f"Presenza {tipo_analisi.upper()} su {ruota_verifica}\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Serie Trigger ({base_conteggio} totali)"
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
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {risultato.get('totale_trigger', 0)}"
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
        n_trigger = risultato.get('totale_trigger', 0)
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
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {n_trigger}"
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
# FUNZIONI LOGICHE (INVARIATE DALLA 3.9.3, ECCETTO cerca_numeri che chiama il popup)
# =============================================================================
# (carica_dati, analizza_ruota_verifica, analizza_antecedenti, 
#  aggiorna_risultati_globali, salva_risultati, format_ambo_terno)
# Le includo per completezza.
def carica_dati(file_path, start_date=None, end_date=None):
    try:
        if not os.path.exists(file_path): return None
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        dates, ruote, numeri, seen_rows, fmt_ok = [], [], [], set(), '%Y/%m/%d'
        for line in lines:
            line = line.strip();
            if not line: continue
            parts = line.split();
            if len(parts) < 7: continue
            data_str, ruota_str, nums_orig = parts[0], parts[1].upper(), parts[2:7]
            try: data_dt_val = datetime.datetime.strptime(data_str, fmt_ok); [int(n) for n in nums_orig]
            except ValueError: continue
            if start_date and end_date and (data_dt_val.date() < start_date.date() or data_dt_val.date() > end_date.date()): continue
            key = f"{data_str}_{ruota_str}";
            if key in seen_rows: continue
            seen_rows.add(key); dates.append(data_str); ruote.append(ruota_str); numeri.append(nums_orig)
        if not dates: return None
        df = pd.DataFrame({'Data': dates, 'Ruota': ruote, **{f'Numero{i+1}': [n[i] for n in numeri] for i in range(5)}})
        df['Data'] = pd.to_datetime(df['Data'], format=fmt_ok)
        for col in [f'Numero{i+1}' for i in range(5)]:
             df[col] = df[col].apply(lambda x: str(int(x)).zfill(2) if pd.notna(x) and str(x).isdigit() and 1 <= int(x) <= 90 else pd.NA)
        df.dropna(subset=[f'Numero{i+1}' for i in range(5)], inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)
        return df if not df.empty else None
    except Exception as e: print(f"Errore lettura file {os.path.basename(file_path)}: {e}"); traceback.print_exc(); return None

def analizza_ruota_verifica(df_verifica, date_trigger_sorted, n_estrazioni, nome_ruota_verifica):
    if df_verifica is None or df_verifica.empty: return None, "Df verifica vuoto."
    df_verifica = df_verifica.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    n_trigger = len(date_trigger_sorted); date_series_verifica = df_verifica['Data']
    freq_estratti, freq_ambi, freq_terne, pres_estratti, pres_ambi, pres_terne = {}, {}, {}, {}, {}, {}
    for data_t in date_trigger_sorted:
        try: start_index = date_series_verifica.searchsorted(data_t, side='right')
        except Exception: continue
        if start_index >= len(date_series_verifica): continue
        df_successive = df_verifica.iloc[start_index : start_index + n_estrazioni]
        estratti_unici_finestra, ambi_unici_finestra, terne_unici_finestra = set(), set(), set()
        if not df_successive.empty:
            for _, row in df_successive.iterrows():
                numeri_estratti_row = [row[col] for col in colonne_numeri if pd.notna(row[col])] 
                if not numeri_estratti_row: continue
                numeri_estratti_validi = sorted([str(n).zfill(2) for n in numeri_estratti_row if str(n).isdigit()])
                for num in numeri_estratti_validi: 
                    freq_estratti[num] = freq_estratti.get(num, 0) + 1
                    estratti_unici_finestra.add(num)
                if len(numeri_estratti_validi) >= 2:
                    for ambo in itertools.combinations(numeri_estratti_validi, 2): 
                        ambo_ordinato = tuple(sorted(ambo)) 
                        freq_ambi[ambo_ordinato] = freq_ambi.get(ambo_ordinato, 0) + 1
                        ambi_unici_finestra.add(ambo_ordinato)
                if len(numeri_estratti_validi) >= 3:
                    for terno in itertools.combinations(numeri_estratti_validi, 3): 
                        terno_ordinato = tuple(sorted(terno)) 
                        freq_terne[terno_ordinato] = freq_terne.get(terno_ordinato, 0) + 1
                        terne_unici_finestra.add(terno_ordinato)
        for num in estratti_unici_finestra: pres_estratti[num] = pres_estratti.get(num, 0) + 1
        for ambo in ambi_unici_finestra: pres_ambi[ambo] = pres_ambi.get(ambo, 0) + 1
        for terno in terne_unici_finestra: pres_terne[terno] = pres_terne.get(terno, 0) + 1
    results = {'totale_trigger': n_trigger}
    for tipo, freq_dict, pres_dict in [('estratto', freq_estratti, pres_estratti), 
                                       ('ambo', freq_ambi, pres_ambi), 
                                       ('terno', freq_terne, pres_terne)]:
        if not freq_dict: 
            results[tipo] = {'presenza': {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'frequenze':pd.Series(dtype=int),'perc_frequenza':pd.Series(dtype=float)},
                             'frequenza':{'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'presenze':pd.Series(dtype=int),'perc_presenza':pd.Series(dtype=float)},
                             'all_percentuali_presenza':pd.Series(dtype=float), 'all_percentuali_frequenza':pd.Series(dtype=float), 
                             'full_presenze':pd.Series(dtype=int), 'full_frequenze':pd.Series(dtype=int)}
            continue
        if tipo in ['ambo', 'terno']:
            freq_s = pd.Series({format_ambo_terno(k): v for k,v in freq_dict.items()}, dtype=int).sort_index()
            pres_s = pd.Series({format_ambo_terno(k): v for k,v in pres_dict.items()}, dtype=int)
        else: 
            freq_s = pd.Series(freq_dict, dtype=int).sort_index()
            pres_s = pd.Series(pres_dict, dtype=int)
        pres_s = pres_s.reindex(freq_s.index, fill_value=0).sort_index()
        tot_freq = freq_s.sum(); perc_freq = (freq_s/tot_freq*100).round(2) if tot_freq > 0 else pd.Series(0.0, index=freq_s.index, dtype=float)
        perc_pres = (pres_s/n_trigger*100).round(2) if n_trigger > 0 else pd.Series(0.0, index=pres_s.index, dtype=float)
        top_pres = pres_s.sort_values(ascending=False).head(10); top_freq = freq_s.sort_values(ascending=False).head(10)
        results[tipo] = {
            'presenza': {'top':top_pres, 'percentuali':perc_pres.reindex(top_pres.index).fillna(0.0), 
                         'frequenze':freq_s.reindex(top_pres.index).fillna(0).astype(int), 
                         'perc_frequenza':perc_freq.reindex(top_pres.index).fillna(0.0)},
            'frequenza':{'top':top_freq, 'percentuali':perc_freq.reindex(top_freq.index).fillna(0.0), 
                         'presenze':pres_s.reindex(top_freq.index).fillna(0).astype(int), 
                         'perc_presenza':perc_pres.reindex(top_freq.index).fillna(0.0)},
            'all_percentuali_presenza':perc_pres, 'all_percentuali_frequenza':perc_freq, 
            'full_presenze':pres_s, 'full_frequenze':freq_s
        }
    return (results, None) if results.get('estratto') or results.get('ambo') or results.get('terno') else (None, f"Nessun risultato su {nome_ruota_verifica}.")

def analizza_antecedenti(df_ruota, numeri_obiettivo, n_precedenti, nome_ruota):
    if df_ruota is None or df_ruota.empty: return None, "DataFrame vuoto."
    if not numeri_obiettivo or n_precedenti <= 0: return None, "Input invalidi."
    df_ruota = df_ruota.sort_values(by='Data').reset_index(drop=True); cols_num = [f'Numero{i+1}' for i in range(5)]
    indices_obiettivo = df_ruota.index[df_ruota[cols_num].isin(numeri_obiettivo).any(axis=1)].tolist()
    n_occ_obiettivo = len(indices_obiettivo)
    if n_occ_obiettivo == 0: return None, f"Obiettivi non trovati su {nome_ruota}."
    freq_ant, pres_ant, actual_base_pres = {}, {}, 0
    for idx_obj in indices_obiettivo:
        if idx_obj < n_precedenti: continue
        actual_base_pres += 1
        df_prec = df_ruota.iloc[idx_obj - n_precedenti : idx_obj]
        if not df_prec.empty:
            numeri_finestra = df_prec[cols_num].values.flatten(); numeri_unici_finestra = set()
            for num in numeri_finestra:
                if pd.notna(num): freq_ant[num] = freq_ant.get(num,0)+1; numeri_unici_finestra.add(num)
            for num_u in numeri_unici_finestra: pres_ant[num_u] = pres_ant.get(num_u,0)+1
    empty_stats = lambda: {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'frequenze':pd.Series(dtype=int),'perc_frequenza':pd.Series(dtype=float)}
    empty_freq_stats = lambda: {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'presenze':pd.Series(dtype=int),'perc_presenza':pd.Series(dtype=float)}
    base_res = {'totale_occorrenze_obiettivo':n_occ_obiettivo, 'base_presenza_antecedenti':actual_base_pres, 'numeri_obiettivo':numeri_obiettivo, 'n_precedenti':n_precedenti, 'nome_ruota':nome_ruota}
    if actual_base_pres == 0 or not freq_ant: return {**base_res, 'presenza':empty_stats(), 'frequenza':empty_freq_stats()}, "Nessuna finestra/numero antecedente valido."
    ant_freq_s = pd.Series(freq_ant, dtype=int).sort_index(); ant_pres_s = pd.Series(pres_ant, dtype=int).reindex(ant_freq_s.index, fill_value=0).sort_index()
    tot_ant_freq = ant_freq_s.sum(); perc_ant_freq = (ant_freq_s/tot_ant_freq*100).round(2) if tot_ant_freq > 0 else pd.Series(0.0, index=ant_freq_s.index)
    perc_ant_pres = (ant_pres_s/actual_base_pres*100).round(2) if actual_base_pres > 0 else pd.Series(0.0, index=ant_pres_s.index)
    top_ant_pres = ant_pres_s.sort_values(ascending=False).head(10); top_ant_freq = ant_freq_s.sort_values(ascending=False).head(10)
    return {**base_res,
            'presenza': {'top':top_ant_pres, 'percentuali':perc_ant_pres.reindex(top_ant_pres.index).fillna(0.0), 'frequenze':ant_freq_s.reindex(top_ant_pres.index).fillna(0).astype(int), 'perc_frequenza':perc_ant_freq.reindex(top_ant_pres.index).fillna(0.0)},
            'frequenza':{'top':top_ant_freq, 'percentuali':perc_ant_freq.reindex(top_ant_freq.index).fillna(0.0), 'presenze':ant_pres_s.reindex(top_ant_freq.index).fillna(0).astype(int), 'perc_presenza':perc_ant_pres.reindex(top_ant_freq.index).fillna(0.0)}
           }, None

def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
    global risultati_globali, info_ricerca_globale
    global button_visualizza, button_verifica_futura, button_verifica_mista 
    # Rimosso button_verifica_esiti
    
    if button_visualizza: button_visualizza.config(state=tk.DISABLED)
    if button_verifica_futura: button_verifica_futura.config(state=tk.DISABLED)
    if button_verifica_mista: button_verifica_mista.config(state=tk.DISABLED)
    
    if modalita == "successivi":
        risultati_globali = risultati_nuovi if risultati_nuovi is not None else []
        info_ricerca_globale = info_ricerca if info_ricerca is not None else {}
        has_valid_results = bool(risultati_globali) and any(res[2] for res in risultati_globali if len(res)>2)
        has_date_trigger = bool(info_ricerca_globale.get('date_trigger_ordinate'))
        has_end_date = info_ricerca_globale.get('end_date') is not None
        has_ruote_verifica_info = bool(info_ricerca_globale.get('ruote_verifica'))
        
        if has_valid_results and button_visualizza: button_visualizza.config(state=tk.NORMAL)
        if has_end_date and has_ruote_verifica_info and button_verifica_futura:
             button_verifica_futura.config(state=tk.NORMAL)
        if has_date_trigger and has_ruote_verifica_info and button_verifica_mista: 
            button_verifica_mista.config(state=tk.NORMAL)
    else: 
        risultati_globali, info_ricerca_globale = [], {}


def salva_risultati():
    global risultato_text
    risultato_text.config(state=tk.NORMAL); content = risultato_text.get(1.0,tk.END).strip(); risultato_text.config(state=tk.DISABLED)
    if not content or any(msg in content for msg in ["Benvenuto","Ricerca in corso...","Nessun risultato"]): return
    fpath = filedialog.asksaveasfilename(defaultextension=".txt",filetypes=[("Text files","*.txt")],title="Salva Risultati")
    if fpath:
        try:
            with open(fpath,"w",encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Salvataggio OK",f"Salvati in:\n{fpath}")
        except Exception as e: messagebox.showerror("Errore Salvataggio",f"Errore:\n{e}")

def format_ambo_terno(combinazione):
    if isinstance(combinazione, tuple) or isinstance(combinazione, list):
        return "-".join(map(str, combinazione))
    return str(combinazione) 

# --- FUNZIONE PER POPUP DI TESTO GENERICO (NON MODALE) ---
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

    close_button_frame = ttk.Frame(popup_window)
    close_button_frame.pack(fill=tk.X, pady=(0,10), padx=10, side=tk.BOTTOM) 
    ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()
    
    popup_window.update_idletasks() 
    master_x = root.winfo_x(); master_y = root.winfo_y()
    master_width = root.winfo_width(); master_height = root.winfo_height()
    win_width = popup_window.winfo_width(); win_height = popup_window.winfo_height()
    center_x = master_x + (master_width // 2) - (win_width // 2)
    center_y = master_y + (master_height // 2) - (win_height // 2)
    popup_window.geometry(f"+{center_x}+{center_y}")

# --- POPUP PER RISULTATI ANALISI SPIA (già presente e corretto) ---
def mostra_popup_risultati_spia(info_ricerca, risultati_analisi):
    # ... (corpo della funzione mostra_popup_risultati_spia come nella tua versione 3.9.3, è già non modale)
    # Assicurati che usi scrolledtext.ScrolledText
    popup = tk.Toplevel(root) 
    popup.title("Riepilogo Analisi Numeri Spia")
    popup.geometry("800x600")
    popup.transient(root) 

    text_area_popup = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
    text_area_popup.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text_area_popup.config(state=tk.NORMAL)
    text_area_popup.delete(1.0, tk.END)
    popup_content = ["=== RIEPILOGO ANALISI NUMERI SPIA ===\n"]
    tipo_spia = info_ricerca.get('tipo_spia_usato', 'N/D').upper()
    spia_val = info_ricerca.get('numeri_spia_input', [])
    if isinstance(spia_val, tuple): spia_display = "-".join(map(str, spia_val))
    elif isinstance(spia_val, list): spia_display = ", ".join(map(str, spia_val))
    else: spia_display = str(spia_val)
    popup_content.append(f"Tipo Spia: {tipo_spia} ({spia_display})")
    popup_content.append(f"Ruote Analisi (Spia): {', '.join(info_ricerca.get('ruote_analisi', []))}")
    popup_content.append(f"Ruote Verifica (Esiti): {', '.join(info_ricerca.get('ruote_verifica', []))}")
    start_date_pd = info_ricerca.get('start_date', pd.NaT)
    end_date_pd = info_ricerca.get('end_date', pd.NaT)
    start_date_str = start_date_pd.strftime('%d/%m/%Y') if pd.notna(start_date_pd) else "N/D"
    end_date_str = end_date_pd.strftime('%d/%m/%Y') if pd.notna(end_date_pd) else "N/D"
    popup_content.append(f"Periodo: {start_date_str} - {end_date_str}")
    popup_content.append(f"Estrazioni Successive Analizzate: {info_ricerca.get('n_estrazioni', 'N/D')}")
    date_triggers = info_ricerca.get('date_trigger_ordinate', [])
    popup_content.append(f"Numero Totale di Eventi Spia (Trigger): {len(date_triggers)}")
    popup_content.append("-" * 50)
    if not risultati_analisi:
        popup_content.append("\nNessun risultato dettagliato per le ruote di verifica.")
    else:
        for nome_ruota_v, _, res_ruota in risultati_analisi:
            if not res_ruota or not isinstance(res_ruota, dict): continue
            popup_content.append(f"\n\n--- RISULTATI PER RUOTA DI VERIFICA: {nome_ruota_v.upper()} ---")
            popup_content.append(f"(Basato su {res_ruota.get('totale_trigger', 0)} eventi spia)")
            for tipo_esito in ['estratto', 'ambo', 'terno']:
                dati_esito = res_ruota.get(tipo_esito)
                if dati_esito:
                    popup_content.append(f"\n  -- {tipo_esito.capitalize()} Successivi --")
                    popup_content.append("    Top per Presenza (su casi trigger):")
                    top_pres = dati_esito.get('presenza', {}).get('top')
                    if top_pres is not None and not top_pres.empty:
                        for item, pres_val in top_pres.items():
                            item_f = format_ambo_terno(item) 
                            perc = dati_esito['presenza']['percentuali'].get(item, 0.0)
                            freq = dati_esito['presenza']['frequenze'].get(item, 0)
                            popup_content.append(f"      - {item_f}: {pres_val} ({perc:.1f}%) [Freq.Tot: {freq}]")
                    else: popup_content.append("      Nessuno.")
                    popup_content.append("    Top per Frequenza Totale:")
                    top_freq = dati_esito.get('frequenza', {}).get('top')
                    if top_freq is not None and not top_freq.empty:
                        for item, freq_val in top_freq.items():
                            item_f = format_ambo_terno(item)
                            perc = dati_esito['frequenza']['percentuali'].get(item, 0.0)
                            pres = dati_esito['frequenza']['presenze'].get(item, 0)
                            popup_content.append(f"      - {item_f}: {freq_val} ({perc:.1f}%) [Pres. su Trigger: {pres}]")
                    else: popup_content.append("      Nessuno.")
                else: popup_content.append(f"\n  -- {tipo_esito.capitalize()} Successivi: Nessun dato trovato.")
    statistiche_combinate_dett = info_ricerca.get('statistiche_combinate_dettagliate') 
    if statistiche_combinate_dett and any(v for v in statistiche_combinate_dett.values() if v):
        popup_content.append("\n\n" + "=" * 20 + " RISULTATI COMBINATI (TUTTE LE RUOTE DI VERIFICA) " + "=" * 20)
        for tipo_esito_comb in ['estratto', 'ambo', 'terno']:
            dati_tipo_comb_dett = statistiche_combinate_dett.get(tipo_esito_comb)
            if dati_tipo_comb_dett: 
                popup_content.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati (per Punteggio) --")
                for i, stat_item in enumerate(dati_tipo_comb_dett): 
                    item_str_c = stat_item["item"]
                    score_c = stat_item["punteggio"]
                    pres_avg_c = stat_item["presenza_media_perc"]
                    freq_tot_c = stat_item["frequenza_totale"]
                    popup_content.append(f"    {i+1}. {item_str_c}: Punt={score_c:.2f} (PresAvg:{pres_avg_c:.1f}%, FreqTot:{freq_tot_c})")
            else: popup_content.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati: Nessuno.")
    elif info_ricerca.get('top_combinati') and any(v for v in info_ricerca.get('top_combinati').values() if v) : 
        popup_content.append("\n\n" + "=" * 20 + " RISULTATI COMBINATI (TUTTE LE RUOTE DI VERIFICA) " + "=" * 20)
        top_combinati_fallback = info_ricerca.get('top_combinati')
        for tipo_esito_comb in ['estratto', 'ambo', 'terno']:
            if top_combinati_fallback.get(tipo_esito_comb):
                popup_content.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati (solo item) --")
                for item_comb in top_combinati_fallback[tipo_esito_comb][:5]: 
                     item_str_c = format_ambo_terno(item_comb) 
                     popup_content.append(f"    - {item_str_c}")
            else: popup_content.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati: Nessuno.")
    text_area_popup.insert(tk.END, "\n".join(popup_content)); text_area_popup.config(state=tk.DISABLED)
    ttk.Button(popup, text="Chiudi", command=popup.destroy).pack(pady=10)


# Modifica a cerca_numeri (invariata rispetto a 3.9.3, ma la includo per completezza)
def cerca_numeri(modalita="successivi"):
    global risultati_globali, info_ricerca_globale, file_ruote, risultato_text, root
    global start_date_entry, end_date_entry, listbox_ruote_analisi, listbox_ruote_verifica
    global entry_numeri_spia, estrazioni_entry_succ, listbox_ruote_analisi_ant
    global entry_numeri_obiettivo, estrazioni_entry_ant, tipo_spia_var_global

    if not mappa_file_ruote() or not file_ruote: 
        messagebox.showerror("Errore File","Mappa file ruote fallita.")
        return
        
    risultati_globali,info_ricerca_globale = [],{}
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END,f"Ricerca {modalita} in corso...\nAttendere prego.\n")
    risultato_text.see(tk.END)
    root.update_idletasks() # Forza aggiornamento GUI per mostrare messaggio
    
    aggiorna_risultati_globali([],{},modalita=modalita) # Resetta e disabilita pulsanti
    
    try:
        start_dt,end_dt = start_date_entry.get_date(),end_date_entry.get_date()
        if start_dt > end_dt: raise ValueError("Data inizio dopo data fine.")
        start_ts,end_ts = pd.Timestamp(start_dt),pd.Timestamp(end_dt)
    except Exception as e: 
        messagebox.showerror("Input Date",f"Date non valide: {e}")
        risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Errore input date."); risultato_text.config(state=tk.DISABLED)
        return
        
    messaggi_out,ris_graf_loc = [],[]
    col_num = [f'Numero{i+1}' for i in range(5)]

    if modalita == "successivi":
        ra_idx,rv_idx = listbox_ruote_analisi.curselection(),listbox_ruote_verifica.curselection()
        if not ra_idx or not rv_idx: 
            messagebox.showwarning("Manca Input","Seleziona Ruote Analisi (Spia) e Ruote Verifica (Esiti).")
            risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
            return
            
        nomi_ra,nomi_rv = [listbox_ruote_analisi.get(i) for i in ra_idx],[listbox_ruote_verifica.get(i) for i in rv_idx]
        
        tipo_spia_scelto = tipo_spia_var_global.get() if tipo_spia_var_global else "estratto"
        
        input_numeri_spia_validi = []
        for entry_widget in entry_numeri_spia:
            val = entry_widget.get().strip()
            if val and val.isdigit() and 1 <= int(val) <= 90:
                input_numeri_spia_validi.append(str(int(val)).zfill(2))
        
        numeri_spia_da_usare = None 
        
        if tipo_spia_scelto == "estratto":
            if not input_numeri_spia_validi:
                messagebox.showwarning("Manca Input","Nessun Numero Spia (Estratto) valido inserito.")
                risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
                return
            numeri_spia_da_usare = input_numeri_spia_validi 
            spia_display_str = ", ".join(numeri_spia_da_usare)
        elif tipo_spia_scelto == "ambo":
            if len(input_numeri_spia_validi) < 2:
                messagebox.showwarning("Manca Input","Inserire almeno 2 numeri per Ambo Spia.")
                risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
                return
            numeri_spia_da_usare = tuple(sorted(input_numeri_spia_validi[:2])) 
            spia_display_str = "-".join(numeri_spia_da_usare)
        else: 
            if not input_numeri_spia_validi:
                messagebox.showwarning("Manca Input","Nessun Numero Spia valido inserito.")
                risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
                return
            numeri_spia_da_usare = input_numeri_spia_validi
            spia_display_str = ", ".join(numeri_spia_da_usare)

        messaggi_out.append(f"Tipo Spia Analizzata: {tipo_spia_scelto.upper()} ({spia_display_str})")

        try: n_estr=int(estrazioni_entry_succ.get()); assert 1<=n_estr<=18
        except: 
            messagebox.showerror("Input Invalido","N. Estrazioni (1-18) non valido.")
            risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input non valido."); risultato_text.config(state=tk.DISABLED)
            return
        
        info_curr={'numeri_spia_input':numeri_spia_da_usare, 
                   'tipo_spia_usato': tipo_spia_scelto,
                   'ruote_analisi':nomi_ra,
                   'ruote_verifica':nomi_rv,
                   'n_estrazioni':n_estr,
                   'start_date':start_ts,
                   'end_date':end_ts}
        
        all_date_trig=set(); messaggi_out.append("\n--- FASE 1: Ricerca Date Uscita Spia ---")
        for nome_ra in nomi_ra:
            df_an=carica_dati(file_ruote.get(nome_ra),start_ts,end_ts)
            if df_an is None or df_an.empty: messaggi_out.append(f"[{nome_ra}] No dati An."); continue
            
            dates_found_this_ruota = []
            if tipo_spia_scelto == "estratto": 
                dates_found_arr=df_an.loc[df_an[col_num].isin(numeri_spia_da_usare).any(axis=1),'Data'].unique()
                dates_found_this_ruota = pd.to_datetime(dates_found_arr).tolist()
            elif tipo_spia_scelto == "ambo": 
                spia_n1, spia_n2 = numeri_spia_da_usare
                for index, row in df_an.iterrows():
                    estratti_riga = set(row[col_num].dropna().tolist())
                    if spia_n1 in estratti_riga and spia_n2 in estratti_riga:
                        dates_found_this_ruota.append(row['Data'])
            
            if dates_found_this_ruota: 
                all_date_trig.update(dates_found_this_ruota)
                messaggi_out.append(f"[{nome_ra}] Trovate {len(dates_found_this_ruota)} date trigger per spia {spia_display_str}.")
            else: messaggi_out.append(f"[{nome_ra}] Nessuna uscita spia {spia_display_str}.")
        
        if not all_date_trig: 
            messaggi_out.append(f"\nNESSUNA USCITA SPIA TROVATA PER {spia_display_str}.")
            aggiorna_risultati_globali([],info_curr,modalita=modalita) # Passa info_curr anche se vuoto
            final_output_no_trigger = "\n".join(messaggi_out)
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END)
            risultato_text.insert(tk.END,final_output_no_trigger); risultato_text.config(state=tk.DISABLED); risultato_text.see("1.0")
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali) 
            return 

        date_trig_ord=sorted(list(all_date_trig)); n_trig_tot=len(date_trig_ord)
        messaggi_out.append(f"\nFASE 1 OK: {n_trig_tot} date trigger totali per spia {spia_display_str}."); info_curr['date_trigger_ordinate']=date_trig_ord
        messaggi_out.append("\n--- FASE 2: Analisi Ruote Verifica ---"); df_cache_ver={}; num_rv_ok=0
        for nome_rv in nomi_rv:
            df_ver_full = df_cache_ver.get(nome_rv)
            if df_ver_full is None: df_ver_full=carica_dati(file_ruote.get(nome_rv),start_ts,end_ts); df_cache_ver[nome_rv]=df_ver_full
            if df_ver_full is None or df_ver_full.empty: messaggi_out.append(f"[{nome_rv}] No dati Ver."); continue
            res_ver,err_ver=analizza_ruota_verifica(df_ver_full,date_trig_ord,n_estr,nome_rv)
            if err_ver: messaggi_out.append(f"[{nome_rv}] Errore: {err_ver}"); continue
            if res_ver:
                ris_graf_loc.append((nome_rv, spia_display_str, res_ver)); num_rv_ok+=1 
                msg_res_v=f"\n=== Risultati Verifica: {nome_rv} (Base: {res_ver['totale_trigger']} trigger) ==="
                for tipo_s_out in ['estratto','ambo','terno']: 
                    res_s_out=res_ver.get(tipo_s_out)
                    if res_s_out:
                        msg_res_v+=f"\n--- {tipo_s_out.capitalize()} Successivi ---\n  Top 10 per Presenza (su {res_ver['totale_trigger']} casi trigger):\n"
                        if not res_s_out['presenza']['top'].empty:
                            for i,(item,pres) in enumerate(res_s_out['presenza']['top'].items()):
                                item_str_out=format_ambo_terno(item)
                                perc_p_out=res_s_out['presenza']['percentuali'].get(item,0.0); freq_p_out=res_s_out['presenza']['frequenze'].get(item,0)
                                msg_res_v+=f"    {i+1}. {item_str_out}: Pres. {pres} ({perc_p_out:.1f}%) | Freq.Tot: {freq_p_out}\n"
                        else: msg_res_v+="    Nessuno.\n"
                        msg_res_v+=f"  Top 10 per Frequenza Totale:\n"
                        if not res_s_out['frequenza']['top'].empty:
                            for i,(item,freq) in enumerate(res_s_out['frequenza']['top'].items()):
                                item_str_out=format_ambo_terno(item)
                                perc_f_out=res_s_out['frequenza']['percentuali'].get(item,0.0); pres_f_out=res_s_out['frequenza']['presenze'].get(item,0)
                                msg_res_v+=f"    {i+1}. {item_str_out}: Freq.Tot: {freq} ({perc_f_out:.1f}%) | Pres. su Trigger: {pres_f_out}\n"
                        else: msg_res_v+="    Nessuno.\n"
                    else: msg_res_v+=f"\n--- {tipo_s_out.capitalize()} Successivi: Nessun risultato ---\n"
                messaggi_out.append(msg_res_v)
            messaggi_out.append("- "*20)
        
        if ris_graf_loc and num_rv_ok > 0:
            messaggi_out.append("\n\n=== RISULTATI COMBINATI (Tutte Ruote Verifica Valide) ===")
            info_curr['statistiche_combinate_dettagliate'] = {} 
            top_comb_ver={'estratto':[],'ambo':[],'terno':[]}; peso_pres,peso_freq=0.6,0.4
            for tipo_comb in ['estratto','ambo','terno']:
                messaggi_out.append(f"\n--- Combinati: {tipo_comb.upper()} Successivi ---"); comb_pres_dict,comb_freq_dict,has_data_comb={},{},False
                for _,_,res_comb in ris_graf_loc:
                    if res_comb and res_comb.get(tipo_comb):
                        has_data_comb=True
                        for item_c,count_c in res_comb[tipo_comb].get('full_presenze',pd.Series(dtype=int)).items(): comb_pres_dict[item_c]=comb_pres_dict.get(item_c,0)+count_c
                        for item_c,count_c in res_comb[tipo_comb].get('full_frequenze',pd.Series(dtype=int)).items(): comb_freq_dict[item_c]=comb_freq_dict.get(item_c,0)+count_c
                if not has_data_comb: messaggi_out.append(f"    Nessun risultato combinato per {tipo_comb}.\n"); continue
                
                comb_pres_s_orig_keys = pd.Series(comb_pres_dict, dtype=int)
                comb_freq_s_orig_keys = pd.Series(comb_freq_dict, dtype=int)
                all_items_idx_comb_orig_keys = comb_pres_s_orig_keys.index.union(comb_freq_s_orig_keys.index)
                def get_sortable_key_comb(item): 
                    if isinstance(item, tuple): return tuple(map(str, item)) 
                    return str(item)
                sortable_index_list_comb = sorted(list(all_items_idx_comb_orig_keys), key=get_sortable_key_comb)
                ordered_index_comb = pd.Index(sortable_index_list_comb)
                comb_pres_s = comb_pres_s_orig_keys.reindex(ordered_index_comb, fill_value=0)
                comb_freq_s = comb_freq_s_orig_keys.reindex(ordered_index_comb, fill_value=0)
                
                tot_pres_ops_comb=n_trig_tot*num_rv_ok
                comb_perc_pres_s=(comb_pres_s/tot_pres_ops_comb*100).round(2) if tot_pres_ops_comb>0 else pd.Series(0.0,index=comb_pres_s.index, dtype=float)
                max_freq_comb=comb_freq_s.max()
                comb_freq_norm_s=(comb_freq_s/max_freq_comb*100).round(2) if max_freq_comb>0 else pd.Series(0.0,index=comb_freq_s.index, dtype=float)
                punt_comb_s=((peso_pres*comb_perc_pres_s)+(peso_freq*comb_freq_norm_s)).round(2).sort_values(ascending=False)
                top_punt_comb=punt_comb_s.head(10)
                
                if not top_punt_comb.empty: 
                    top_comb_ver[tipo_comb]=top_punt_comb.index.tolist()[:5] 
                    stat_dett_comb = []
                    for item_idx, score_val in top_punt_comb.items(): 
                        pres_avg_val = comb_perc_pres_s.get(item_idx, 0.0)
                        freq_tot_val = comb_freq_s.get(item_idx, 0)
                        stat_dett_comb.append({
                           "item": item_idx, 
                           "punteggio": score_val,
                           "presenza_media_perc": pres_avg_val,
                           "frequenza_totale": freq_tot_val
                        })
                    info_curr.setdefault('statistiche_combinate_dettagliate', {})[tipo_comb] = stat_dett_comb
                messaggi_out.append(f"  Top 10 Combinati per Punteggio:\n")
                if not top_punt_comb.empty:
                    for i,(item_comb_idx,score_comb) in enumerate(top_punt_comb.items()):
                        messaggi_out.append(f"    {i+1}. {item_comb_idx}: Punt={score_comb:.2f} (PresAvg:{comb_perc_pres_s.get(item_comb_idx,0.0):.1f}%, FreqTot:{comb_freq_s.get(item_comb_idx,0)})\n")
                else: messaggi_out.append("    Nessuno.\n")
            info_curr['top_combinati']=top_comb_ver
        elif num_rv_ok == 0: messaggi_out.append("\nNessuna Ruota Verifica valida con risultati.")
        aggiorna_risultati_globali(ris_graf_loc,info_curr,modalita="successivi")
        
        if ris_graf_loc or (not all_date_trig and tipo_spia_scelto): 
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali)
        
    elif modalita == "antecedenti":
        ra_ant_idx=listbox_ruote_analisi_ant.curselection()
        if not ra_ant_idx: 
            messagebox.showwarning("Manca Input","Seleziona Ruota/e Analisi.")
            risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
            return
        nomi_ra_ant=[listbox_ruote_analisi_ant.get(i) for i in ra_ant_idx]
        num_obj=sorted(list(set(str(int(e.get().strip())).zfill(2) for e in entry_numeri_obiettivo if e.get().strip() and e.get().strip().isdigit() and 1<=int(e.get().strip())<=90)))
        if not num_obj: 
            messagebox.showwarning("Manca Input","Numeri Obiettivo non validi.")
            risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input mancante."); risultato_text.config(state=tk.DISABLED)
            return
        try: n_prec=int(estrazioni_entry_ant.get()); assert n_prec >=1
        except: 
            messagebox.showerror("Input Invalido","N. Precedenti (>=1) non valido.")
            risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Input non valido."); risultato_text.config(state=tk.DISABLED)
            return
        
        messaggi_out.append(f"--- Analisi Antecedenti (Marker) ---")
        messaggi_out.append(f"Numeri Obiettivo: {', '.join(num_obj)}")
        messaggi_out.append(f"Numero Estrazioni Precedenti Controllate: {n_prec}")
        messaggi_out.append(f"Periodo: {start_ts.strftime('%d/%m/%Y')} - {end_ts.strftime('%d/%m/%Y')}")
        messaggi_out.append("-" * 40)

        df_cache_ant={}; almeno_un_risultato_antecedente = False

        for nome_ra_ant in nomi_ra_ant:
            df_ant_full = df_cache_ant.get(nome_ra_ant)
            if df_ant_full is None: 
                df_ant_full=carica_dati(file_ruote.get(nome_ra_ant),start_ts,end_ts)
                df_cache_ant[nome_ra_ant]=df_ant_full
            
            if df_ant_full is None or df_ant_full.empty: 
                messaggi_out.append(f"\n[{nome_ra_ant.upper()}] Nessun dato storico trovato per il periodo selezionato."); 
                continue

            res_ant,err_ant=analizza_antecedenti(df_ruota=df_ant_full, numeri_obiettivo=num_obj, n_precedenti=n_prec, nome_ruota=nome_ra_ant)
            
            if err_ant: 
                messaggi_out.append(f"\n[{nome_ra_ant.upper()}] Errore: {err_ant}"); 
                continue

            if res_ant and res_ant.get('base_presenza_antecedenti',0)>0 and \
               ( (res_ant.get('presenza') and not res_ant['presenza']['top'].empty) or \
                 (res_ant.get('frequenza') and not res_ant['frequenza']['top'].empty) ):
                almeno_un_risultato_antecedente = True
                msg_res_ant=f"\n=== Risultati Antecedenti per Ruota: {nome_ra_ant.upper()} ==="
                msg_res_ant+=f"\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Estrazioni Prec.: {res_ant['n_precedenti']} | Occorrenze Obiettivo: {res_ant['totale_occorrenze_obiettivo']})"
                
                if res_ant.get('presenza') and not res_ant['presenza']['top'].empty:
                    msg_res_ant+=f"\n  Top Antecedenti per Presenza (su {res_ant['base_presenza_antecedenti']} casi validi):"
                    for i,(num,pres) in enumerate(res_ant['presenza']['top'].head(10).items()): 
                        perc_pres_val = res_ant['presenza']['percentuali'].get(num,0.0)
                        freq_val = res_ant['presenza']['frequenze'].get(num,0)
                        msg_res_ant+=f"\n    {i+1}. {num}: {pres} ({perc_pres_val:.1f}%) [Freq.Tot: {freq_val}]"
                else: msg_res_ant+="\n  Nessun Top per Presenza."
                
                if res_ant.get('frequenza') and not res_ant['frequenza']['top'].empty:
                    msg_res_ant+=f"\n  Top Antecedenti per Frequenza Totale:"
                    for i,(num,freq) in enumerate(res_ant['frequenza']['top'].head(10).items()):
                        perc_freq_val = res_ant['frequenza']['percentuali'].get(num,0.0)
                        pres_val = res_ant['frequenza']['presenze'].get(num,0)
                        msg_res_ant+=f"\n    {i+1}. {num}: {freq} ({perc_freq_val:.1f}%) [Pres. su Casi: {pres_val}]"
                else: msg_res_ant+="\n  Nessun Top per Frequenza."
                messaggi_out.append(msg_res_ant)
            else: 
                messaggi_out.append(f"\n[{nome_ra_ant.upper()}] Nessun dato antecedente significativo trovato per gli obiettivi specificati.")
            messaggi_out.append("\n" + ("- "*20))
        
        aggiorna_risultati_globali([],{},modalita="antecedenti") 
        
    final_output="\n".join(messaggi_out) if messaggi_out else "Nessun risultato."
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,final_output); risultato_text.config(state=tk.DISABLED); risultato_text.see("1.0")

    if modalita == "antecedenti" and almeno_un_risultato_antecedente:
        mostra_popup_testo_semplice("Riepilogo Analisi Numeri Antecedenti", final_output)


# =============================================================================
# FUNZIONI PER VERIFICA ESITI
# =============================================================================
def verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_ruote_verifica, n_verifiche, start_ts, end_ts, titolo_sezione="VERIFICA MISTA SU TRIGGER"):
    if not date_triggers or not combinazioni_utente or not nomi_ruote_verifica: 
        return "Errore: Dati input mancanti per verifica utente su triggers."
    
    estratti_u = sorted(list(set(combinazioni_utente.get('estratto', []))))
    ambi_u_tpl = sorted(list(set(tuple(sorted(a)) for a in combinazioni_utente.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_u_tpl = sorted(list(set(tuple(sorted(t)) for t in combinazioni_utente.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
    quaterne_u_tpl = sorted(list(set(tuple(sorted(q)) for q in combinazioni_utente.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_u_tpl = sorted(list(set(tuple(sorted(c)) for c in combinazioni_utente.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))

    # Dizionario per tracciare gli esiti per ogni item, ruota e colpo
    # Struttura: hits_per_item[tipo_sorte][item_utente] = [(ruota, colpo, data_estrazione), ...]
    hits_per_item = {
        'estratto': {e: [] for e in estratti_u}, 
        'ambo': {a: [] for a in ambi_u_tpl}, 
        'terno': {t: [] for t in terni_u_tpl},
        'quaterna': {q: [] for q in quaterne_u_tpl},
        'cinquina': {c: [] for c in cinquine_u_tpl}
    }
    
    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver = {}; ruote_valide = []
    for nome_rv_loop in nomi_ruote_verifica:
        df_ver = carica_dati(file_ruote.get(nome_rv_loop), start_date=start_ts, end_date=None) 
        if df_ver is not None and not df_ver.empty: 
            df_cache_ver[nome_rv_loop] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
            ruote_valide.append(nome_rv_loop)
            
    if not ruote_valide: return "Errore: Nessuna ruota di verifica valida per caricare i dati del periodo."
    
    casi_tot_eff_per_ruota = len(date_triggers) # Il numero di volte che ogni ruota di verifica viene controllata
    if casi_tot_eff_per_ruota == 0: return "Nessun caso trigger da verificare."

    for trigger_idx, data_t in enumerate(date_triggers): 
        for nome_rv in ruote_valide:
            df_v = df_cache_ver.get(nome_rv)
            if df_v is None: continue
            date_s_v = df_v['Data']
            try: start_idx = date_s_v.searchsorted(data_t, side='right')
            except Exception: continue
            if start_idx >= len(date_s_v): continue
            
            df_fin_v = df_v.iloc[start_idx : start_idx + n_verifiche] 
            
            if not df_fin_v.empty:
                for colpo_idx, (_, row) in enumerate(df_fin_v.iterrows(), 1):
                    data_estrazione_corrente = row['Data'].date()
                    current_row_numbers_v = [row[col] for col in cols_num if pd.notna(row[col])]
                    nums_draw = sorted(current_row_numbers_v) 
                    if not nums_draw: continue 
                    set_nums_draw = set(nums_draw)
                    
                    if estratti_u:
                        for num_hit in set_nums_draw.intersection(set(estratti_u)):
                            hits_per_item['estratto'][num_hit].append((nome_rv, colpo_idx, data_estrazione_corrente))
                    if ambi_u_tpl and len(nums_draw) >= 2:
                        ambi_generati_da_riga = set(itertools.combinations(nums_draw, 2))
                        for ambo_hit in ambi_generati_da_riga.intersection(set(ambi_u_tpl)):
                            hits_per_item['ambo'][ambo_hit].append((nome_rv, colpo_idx, data_estrazione_corrente))
                    if terni_u_tpl and len(nums_draw) >= 3:
                        terni_generati_da_riga = set(itertools.combinations(nums_draw, 3))
                        for terno_hit in terni_generati_da_riga.intersection(set(terni_u_tpl)): 
                            hits_per_item['terno'][terno_hit].append((nome_rv, colpo_idx, data_estrazione_corrente))
                    if quaterne_u_tpl and len(nums_draw) >= 4:
                        quaterne_generati_da_riga = set(itertools.combinations(nums_draw, 4))
                        for quaterna_hit in quaterne_generati_da_riga.intersection(set(quaterne_u_tpl)):
                            hits_per_item['quaterna'][quaterna_hit].append((nome_rv, colpo_idx, data_estrazione_corrente))
                    if cinquine_u_tpl and len(nums_draw) >= 5:
                        cinquine_generati_da_riga = set(itertools.combinations(nums_draw, 5))
                        for cinquina_hit in cinquine_generati_da_riga.intersection(set(cinquine_u_tpl)):
                           hits_per_item['cinquina'][cinquina_hit].append((nome_rv, colpo_idx, data_estrazione_corrente))
    
    out = [f"\n\n=== {titolo_sezione} ({n_verifiche} Colpi dopo ogni Trigger) ==="]
    out.append(f"Numero di casi trigger di base: {casi_tot_eff_per_ruota}")
    out.append(f"Ruote di verifica considerate: {', '.join(ruote_valide) or 'Nessuna'}")

    sorti_verificate = [
        ('estratto', estratti_u), ('ambo', ambi_u_tpl), ('terno', terni_u_tpl),
        ('quaterna', quaterne_u_tpl), ('cinquina', cinquine_u_tpl)
    ]

    for tipo_sorte, items_da_verificare in sorti_verificate:
        if not items_da_verificare: continue
        out.append(f"\n--- Esiti {tipo_sorte.upper()} ---")
        almeno_un_hit_per_sorte_tipo = False
        for item_utente in items_da_verificare:
            item_str = format_ambo_terno(item_utente) if isinstance(item_utente, tuple) else item_utente
            lista_esiti_item = hits_per_item[tipo_sorte].get(item_utente, [])
            
            if lista_esiti_item:
                almeno_un_hit_per_sorte_tipo = True
                # Raggruppa esiti per (ruota, colpo) per evitare duplicati se un trigger appare più volte
                # e poi formatta. Per semplicità, ora mostriamo tutti gli hit individuali.
                # Per una visualizzazione più pulita come quella futura, potremmo voler mostrare solo il primo hit per trigger window.
                # Ma per ora, mostriamo tutti gli hit raggruppati per (ruota, colpo)
                
                esiti_unici_formattati = []
                # Per evitare ripetizioni se lo stesso item esce più volte per lo STESSO trigger window su ruote diverse
                # o in colpi diversi, potremmo voler limitare l'output.
                # L'approccio di "verifica_esiti_futuri" è mostrare il primo colpo/ruota.
                # Qui, dato che i trigger sono multipli, mostrare tutti gli hit può essere informativo.
                
                # Semplifichiamo: mostriamo fino a N dettagli per non sovraffollare.
                dettagli_output = []
                esiti_gia_mostrati_per_trigger_ruota_colpo = set() # Per evitare duplicati nella stessa riga di output

                for ruota_hit, colpo_hit, data_hit in sorted(lista_esiti_item, key=lambda x: (x[1], x[0], x[2])): # Ordina per colpo, poi ruota, poi data
                    chiave_univoca = (trigger_idx, ruota_hit, colpo_hit) # Questo non funziona bene qui perché trigger_idx è fuori scope
                                                                    # Dovremmo forse basarci sulla data_t del trigger?
                                                                    # Per ora, semplifichiamo e mostriamo tutti gli hit.
                    dettagli_output.append(f"{ruota_hit} @ C{colpo_hit} ({data_hit.strftime('%Y-%m-%d')})")

                if dettagli_output:
                     out.append(f"    - {item_str}: USCITO -> {'; '.join(dettagli_output[:10])}{'...' if len(dettagli_output) > 10 else ''}")
                else: # Dovrebbe essere già gestito da if lista_esiti_item
                     out.append(f"    - {item_str}: NON uscito")

            else:
                out.append(f"    - {item_str}: NON uscito")
        
        if not almeno_un_hit_per_sorte_tipo and items_da_verificare:
            out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito.")
            
    return "\n".join(out)

# def verifica_esiti_combinati(date_triggers, top_combinati, nomi_ruote_verifica, n_verifiche, start_ts, end_ts):
#     # Funzione commentata/rimossa come da richiesta
#     pass

# def esegui_verifica_esiti():
#     # Funzione commentata/rimossa come da richiesta
#     pass


def verifica_esiti_futuri(top_combinati_input, nomi_ruote_verifica, data_fine_analisi, n_colpi_futuri):
    if not top_combinati_input or not any(top_combinati_input.values()) or not nomi_ruote_verifica or data_fine_analisi is None or n_colpi_futuri <= 0: return "Errore: Input invalidi per verifica_esiti_futuri (post-analisi)."
    estratti_items = sorted(list(set(top_combinati_input.get('estratto', [])))); ambi_items = sorted(list(set(tuple(sorted(a)) for a in top_combinati_input.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_items = sorted(list(set(tuple(sorted(t)) for t in top_combinati_input.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3))); quaterne_items = sorted(list(set(tuple(sorted(q)) for q in top_combinati_input.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_items = sorted(list(set(tuple(sorted(c)) for c in top_combinati_input.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5))); set_estratti = set(estratti_items); set_ambi = set(ambi_items); set_terni = set(terni_items)
    set_quaterne = set(quaterne_items); set_cinquine = set(cinquine_items); cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver_fut = {}; ruote_con_dati_fut = []
    for nome_rv in nomi_ruote_verifica:
        df_ver_full = carica_dati(file_ruote.get(nome_rv), start_date=None, end_date=None)
        if df_ver_full is None or df_ver_full.empty: continue
        df_ver_fut = df_ver_full[df_ver_full['Data'] > data_fine_analisi].copy().sort_values(by='Data').reset_index(drop=True); df_fin_fut = df_ver_fut.head(n_colpi_futuri)
        if not df_fin_fut.empty: df_cache_ver_fut[nome_rv] = df_fin_fut; ruote_con_dati_fut.append(nome_rv)
    if not ruote_con_dati_fut: return f"Nessuna estrazione trovata su nessuna ruota di verifica dopo {data_fine_analisi.date()} per {n_colpi_futuri} colpi."
    hits_registrati = {'estratto': {e: [] for e in estratti_items}, 'ambo': {a: [] for a in ambi_items},'terno': {t: [] for t in terni_items}, 'quaterna': {q: [] for q in quaterne_items},'cinquina': {c: [] for c in cinquine_items}}
    primo_hit_assoluto = {'estratto': set(), 'ambo': set(), 'terno': set(), 'quaterna': set(), 'cinquina': set()}
    for nome_rv in ruote_con_dati_fut:
        df_finestra_ruota = df_cache_ver_fut[nome_rv]
        for colpo_idx, (_, row) in enumerate(df_finestra_ruota.iterrows(), 1):
            numeri_estratti_riga = [row[col] for col in cols_num if pd.notna(row[col])]; numeri_sortati_riga = sorted(numeri_estratti_riga)
            if not numeri_sortati_riga: continue 
            set_numeri_riga = set(numeri_sortati_riga)
            if estratti_items:
                for item_e in set_numeri_riga.intersection(set_estratti):
                    if ('estratto', item_e) not in primo_hit_assoluto['estratto']: hits_registrati['estratto'][item_e].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['estratto'].add(('estratto', item_e))
            if ambi_items and len(numeri_sortati_riga) >= 2:
                for item_a in set(itertools.combinations(numeri_sortati_riga,2)).intersection(set_ambi):
                    if ('ambo', item_a) not in primo_hit_assoluto['ambo']: hits_registrati['ambo'][item_a].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['ambo'].add(('ambo', item_a))
            if terni_items and len(numeri_sortati_riga) >= 3:
                for item_t in set(itertools.combinations(numeri_sortati_riga,3)).intersection(set_terni):
                    if ('terno', item_t) not in primo_hit_assoluto['terno']: hits_registrati['terno'][item_t].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['terno'].add(('terno', item_t))
            if quaterne_items and len(numeri_sortati_riga) >= 4:
                for item_q in set(itertools.combinations(numeri_sortati_riga,4)).intersection(set_quaterne):
                    if ('quaterna', item_q) not in primo_hit_assoluto['quaterna']: hits_registrati['quaterna'][item_q].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['quaterna'].add(('quaterna', item_q))
            if cinquine_items and len(numeri_sortati_riga) >= 5:
                for item_c in set(itertools.combinations(numeri_sortati_riga,5)).intersection(set_cinquine):
                    if ('cinquina', item_c) not in primo_hit_assoluto['cinquina']: hits_registrati['cinquina'][item_c].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['cinquina'].add(('cinquina', item_c))
    out = [f"\n\n=== VERIFICA ESITI FUTURI (POST-ANALISI) ({n_colpi_futuri} Colpi dopo {data_fine_analisi.date()}) ==="]
    out.append(f"Ruote verificate con dati futuri disponibili: {', '.join(ruote_con_dati_fut) or 'Nessuna'}")
    sorti_config = [('estratto', estratti_items), ('ambo', ambi_items), ('terno', terni_items), ('quaterna', quaterne_items), ('cinquina', cinquine_items)]
    for tipo_sorte, lista_items_sorte in sorti_config:
        if not lista_items_sorte: continue 
        out.append(f"\n--- Esiti Futuri {tipo_sorte.upper()} ---")
        almeno_un_hit_per_sorte = False
        for item_da_verificare in lista_items_sorte:
            item_str_formattato = format_ambo_terno(item_da_verificare) if isinstance(item_da_verificare, tuple) else item_da_verificare
            dettagli_hit_item = hits_registrati[tipo_sorte].get(item_da_verificare, [])
            if dettagli_hit_item:
                almeno_un_hit_per_sorte = True; dettagli_ordinati = sorted(dettagli_hit_item, key=lambda x: (x[1], x[0]))
                stringa_dettagli = "; ".join([f"{d_ruota} @ C{d_colpo} ({d_data})" for d_ruota, d_colpo, d_data in dettagli_ordinati])
                out.append(f"    - {item_str_formattato}: USCITO -> {stringa_dettagli}")
            else: out.append(f"    - {item_str_formattato}: NON uscito")
        if not almeno_un_hit_per_sorte and lista_items_sorte: out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito nei colpi futuri analizzati.")
    return "\n".join(out)

def esegui_verifica_futura():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica_futura
    risultato_text.config(state=tk.NORMAL); risultato_text.insert(tk.END, "\n\nVerifica esiti futuri (post-analisi)..."); risultato_text.see(tk.END); root.update_idletasks()
    top_c = info_ricerca_globale.get('top_combinati'); nomi_rv = info_ricerca_globale.get('ruote_verifica'); data_fine = info_ricerca_globale.get('end_date')
    if not all([top_c, nomi_rv, data_fine]): 
        messagebox.showerror("Errore Verifica Futura", "Dati analisi 'Successivi' (Top combinati, Ruote verifica, Data Fine) mancanti."); 
        risultato_text.config(state=tk.DISABLED); return
    if not any(v for v in top_c.values() if isinstance(v, list) and v): 
        messagebox.showinfo("Verifica Futura", "Nessun 'Top Combinato' dall'analisi precedente da verificare."); 
        risultato_text.config(state=tk.DISABLED); return
    try: n_colpi_fut = int(estrazioni_entry_verifica_futura.get()); assert 1 <= n_colpi_fut <= 50
    except: 
        messagebox.showerror("Input Invalido", "N. Colpi Verifica Futura (1-50) non valido."); 
        risultato_text.config(state=tk.DISABLED); return
    try: 
        res_str = verifica_esiti_futuri(top_c, nomi_rv, data_fine, n_colpi_fut)
        risultato_text.insert(tk.END, res_str)
        if res_str and "Errore" not in res_str and "Nessuna estrazione trovata" not in res_str :
            mostra_popup_testo_semplice("Riepilogo Verifica Predittiva (Post-Analisi)", res_str)
    except Exception as e: 
        risultato_text.insert(tk.END, f"\nErrore durante la verifica esiti futuri: {e}"); traceback.print_exc()
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)

def esegui_verifica_mista():
    global info_ricerca_globale, risultato_text, root, text_combinazioni_miste, estrazioni_entry_verifica_mista
    risultato_text.config(state=tk.NORMAL); risultato_text.insert(tk.END, "\n\nVerifica mista (combinazioni utente su trigger spia)..."); risultato_text.see(tk.END); root.update_idletasks()
    input_text = text_combinazioni_miste.get("1.0", tk.END).strip()
    if not input_text: messagebox.showerror("Input Invalido", "Nessuna combinazione inserita."); risultato_text.config(state=tk.DISABLED); return
    combinazioni_sets = {'estratto': set(),'ambo': set(),'terno': set(),'quaterna': set(),'cinquina': set()}; righe_input_originali = []
    for riga_idx, riga in enumerate(input_text.splitlines()):
        riga_proc = riga.strip()
        if not riga_proc: continue
        righe_input_originali.append(riga_proc)
        try:
            numeri_str = riga_proc.split('-') if '-' in riga_proc else riga_proc.split()
            numeri_int = [int(n.strip()) for n in numeri_str]
            if not all(1<=n<=90 for n in numeri_int): raise ValueError("Numeri fuori range (1-90).")
            if len(set(numeri_int))!=len(numeri_int): raise ValueError("Numeri duplicati.")
            if not (1<=len(numeri_int)<=5): raise ValueError("Inserire da 1 a 5 numeri.")
            numeri_validi_zfill = sorted([str(n).zfill(2) for n in numeri_int])
            for i in range(1, len(numeri_validi_zfill) + 1):
                if i > 5: continue 
                for comb in itertools.combinations(numeri_validi_zfill, i):
                    if i==1: combinazioni_sets['estratto'].add(comb[0])
                    else: combinazioni_sets[['ambo','terno','quaterna','cinquina'][i-2]].add(tuple(sorted(comb)))                   
        except ValueError as ve: messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {ve}"); risultato_text.config(state=tk.DISABLED); return
        except Exception as e_parse: messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {e_parse}"); risultato_text.config(state=tk.DISABLED); return
    combinazioni_utente = {k: sorted(list(v)) for k, v in combinazioni_sets.items()}
    if not any(combinazioni_utente.values()): messagebox.showerror("Input Invalido","Nessuna combinazione valida."); risultato_text.config(state=tk.DISABLED); return
    date_triggers = info_ricerca_globale.get('date_trigger_ordinate'); nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date'); end_ts = info_ricerca_globale.get('end_date')
    numeri_spia_originali = info_ricerca_globale.get('numeri_spia_input', []) 
    if isinstance(numeri_spia_originali, tuple): spia_display_originale = "-".join(numeri_spia_originali)
    elif isinstance(numeri_spia_originali, list): spia_display_originale = ", ".join(numeri_spia_originali)
    else: spia_display_originale = str(numeri_spia_originali)
    if not all([date_triggers, nomi_rv, start_ts, end_ts]): messagebox.showerror("Errore Verifica Mista", "Dati analisi 'Successivi' (Date Trigger, Ruote Verifica, Periodo Analisi) mancanti."); risultato_text.config(state=tk.DISABLED); return
    try: n_colpi_misti = int(estrazioni_entry_verifica_mista.get()); assert 1 <= n_colpi_misti <= 18
    except: messagebox.showerror("Input Invalido", "N. Colpi Verifica Mista (1-18) non valido."); risultato_text.config(state=tk.DISABLED); return
    try:
        titolo_output = f"VERIFICA MISTA (COMBINAZIONI UTENTE) - Dopo Spia: {spia_display_originale}"
        res_str = verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_rv, n_colpi_misti, start_ts, end_ts, titolo_sezione=titolo_output)
        summary_input = "\nInput utente originale:\n" + "\n".join([f"  - {r}" for r in righe_input_originali])
        lines = res_str.splitlines(); insert_idx_summary = 1; final_output_lines = lines[:insert_idx_summary] + [summary_input] + lines[insert_idx_summary:]
        full_res_str_mista = "\n".join(final_output_lines)
        risultato_text.insert(tk.END, full_res_str_mista)
        if full_res_str_mista and "Errore" not in full_res_str_mista and "Nessun caso trigger" not in full_res_str_mista:
            mostra_popup_testo_semplice(f"Riepilogo Verifica Mista (Spia: {spia_display_originale})", full_res_str_mista)
    except Exception as e: risultato_text.insert(tk.END, f"\nErrore durante la verifica mista: {e}"); traceback.print_exc()
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)

# =============================================================================
# Funzione Wrapper per Visualizza Grafici (invariata)
# =============================================================================
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale 
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and bool(risultati_globali) and any(r[2] for r in risultati_globali if len(r)>2):
        valid_res = [r for r in risultati_globali if r[2] is not None]
        if valid_res: visualizza_grafici(valid_res, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni',5))
        else: messagebox.showinfo("Grafici", "Nessun risultato valido per grafici.")
    else: messagebox.showinfo("Grafici", "Esegui 'Cerca Successivi' con risultati validi prima.")

# =============================================================================
# GUI e Mainloop (Modificato testo di benvenuto)
# =============================================================================
root = tk.Tk()
root.title("Analisi Lotto v3.9.5 - Popup Verifiche") 
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

main_frame = ttk.Frame(root, padding=10); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
cartella_frame = ttk.Frame(main_frame); cartella_frame.pack(fill=tk.X, pady=(0,10))
ttk.Label(cartella_frame, text="Cartella Estrazioni:", style="Title.TLabel").pack(side=tk.LEFT, padx=(0,5))
cartella_entry = ttk.Entry(cartella_frame, width=60); cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
btn_sfoglia = ttk.Button(cartella_frame, text="Sfoglia...") 
btn_sfoglia.pack(side=tk.LEFT, padx=5)

notebook = ttk.Notebook(main_frame, style="TNotebook"); notebook.pack(fill=tk.X, pady=10)
tab_successivi = ttk.Frame(notebook, padding=10); notebook.add(tab_successivi, text=' Analisi Numeri Successivi (E/A/T) ')
controls_frame_succ = ttk.Frame(tab_successivi); controls_frame_succ.pack(fill=tk.X)
controls_frame_succ.columnconfigure(0, weight=1); controls_frame_succ.columnconfigure(1, weight=1)

ruote_analisi_outer_frame = ttk.Frame(controls_frame_succ); ruote_analisi_outer_frame.grid(row=0,column=0,sticky="nsew",padx=(0,5))
ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi (Spia):", style="Title.TLabel").pack(anchor="w") 
ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame); ruote_analisi_list_frame.pack(fill=tk.BOTH,expand=True)
scrollbar_ra = ttk.Scrollbar(ruote_analisi_list_frame); scrollbar_ra.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_ra.set)
listbox_ruote_analisi.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_ra.config(command=listbox_ruote_analisi.yview)

ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ); ruote_verifica_outer_frame.grid(row=0,column=1,sticky="nsew",padx=5)
ttk.Label(ruote_verifica_outer_frame, text="4. Ruote Verifica (Esiti):", style="Title.TLabel").pack(anchor="w") 
ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame); ruote_verifica_list_frame.pack(fill=tk.BOTH,expand=True)
scrollbar_rv = ttk.Scrollbar(ruote_verifica_list_frame); scrollbar_rv.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_rv.set)
listbox_ruote_verifica.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_rv.config(command=listbox_ruote_verifica.yview)

center_controls_frame_succ = ttk.Frame(controls_frame_succ); center_controls_frame_succ.grid(row=0,column=2,sticky="ns",padx=5)

tipo_spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Tipo di Spia ", padding=5)
tipo_spia_frame_succ.pack(fill=tk.X, pady=(0,5))
tipo_spia_var_global = tk.StringVar(value="estratto") 
ttk.Radiobutton(tipo_spia_frame_succ, text="Estratto Spia", variable=tipo_spia_var_global, value="estratto", style="TRadiobutton").pack(anchor="w", padx=5)
ttk.Radiobutton(tipo_spia_frame_succ, text="Ambo Spia", variable=tipo_spia_var_global, value="ambo", style="TRadiobutton").pack(anchor="w", padx=5)

spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 3. Numeri Spia (1-5) ",padding=5); spia_frame_succ.pack(fill=tk.X,pady=(0,5)) 
spia_entry_container_succ = ttk.Frame(spia_frame_succ); spia_entry_container_succ.pack(fill=tk.X,pady=5)
entry_numeri_spia = [ttk.Entry(spia_entry_container_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)) for _ in range(5)]
for entry in entry_numeri_spia: entry.pack(side=tk.LEFT,padx=3,ipady=2)

estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Estrazioni Successive ",padding=5); estrazioni_frame_succ.pack(fill=tk.X,pady=5) 
ttk.Label(estrazioni_frame_succ, text="Quante (1-18):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_succ.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_succ.insert(0,"5")

# Frame per la verifica classica (entry disabilitata)
# verifica_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" (Verifica Classica Rimossa) ",padding=5) 
# verifica_frame_succ.pack(fill=tk.X,pady=5) 
# ttk.Label(verifica_frame_succ, text="Estrazioni Verifica (1-18):",style="Small.TLabel").pack(anchor="w")
# estrazioni_entry_verifica = ttk.Entry(verifica_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10), state=tk.DISABLED) 
# estrazioni_entry_verifica.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica.insert(0,"9")


buttons_frame_succ = ttk.Frame(controls_frame_succ); buttons_frame_succ.grid(row=0,column=3,sticky="ns",padx=(10,0))
button_cerca_succ = ttk.Button(buttons_frame_succ, text="Cerca Successivi",command=lambda:cerca_numeri(modalita="successivi")); button_cerca_succ.pack(pady=5,fill=tk.X,ipady=3)
# button_verifica_esiti = ttk.Button(buttons_frame_succ, text="Verifica Esiti\n(Classica)",command=esegui_verifica_esiti); # RIMOSSO
# button_verifica_esiti.pack(pady=5,fill=tk.X,ipady=0); button_verifica_esiti.config(state=tk.DISABLED) # RIMOSSO

tab_antecedenti = ttk.Frame(notebook,padding=10); notebook.add(tab_antecedenti, text=' Analisi Numeri Antecedenti (Marker) ')
controls_frame_ant = ttk.Frame(tab_antecedenti); controls_frame_ant.pack(fill=tk.X)
controls_frame_ant.columnconfigure(0,weight=1)
ruote_analisi_ant_outer_frame = ttk.Frame(controls_frame_ant); ruote_analisi_ant_outer_frame.grid(row=0,column=0,sticky="nsew",padx=(0,10))
ttk.Label(ruote_analisi_ant_outer_frame, text="1. Ruote da Analizzare:",style="Title.TLabel").pack(anchor="w")
ttk.Label(ruote_analisi_ant_outer_frame, text="(Obiettivo e antecedenti cercati qui)",style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_ant_list_frame = ttk.Frame(ruote_analisi_ant_outer_frame); ruote_analisi_ant_list_frame.pack(fill=tk.BOTH,expand=True)
scrollbar_ra_ant = ttk.Scrollbar(ruote_analisi_ant_list_frame); scrollbar_ra_ant.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_analisi_ant = tk.Listbox(ruote_analisi_ant_list_frame,height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_ra_ant.set)
listbox_ruote_analisi_ant.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_ra_ant.config(command=listbox_ruote_analisi_ant.yview)
center_controls_frame_ant = ttk.Frame(controls_frame_ant); center_controls_frame_ant.grid(row=0,column=1,sticky="ns",padx=10)
obiettivo_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 2. Numeri Obiettivo (1-90) ",padding=5); obiettivo_frame_ant.pack(fill=tk.X,pady=(0,5))
obiettivo_entry_container_ant = ttk.Frame(obiettivo_frame_ant); obiettivo_entry_container_ant.pack(fill=tk.X,pady=5)
entry_numeri_obiettivo = [ttk.Entry(obiettivo_entry_container_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10)) for _ in range(5)]
for entry in entry_numeri_obiettivo: entry.pack(side=tk.LEFT,padx=3,ipady=2)
estrazioni_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 3. Estrazioni Precedenti ",padding=5); estrazioni_frame_ant.pack(fill=tk.X,pady=5)
ttk.Label(estrazioni_frame_ant, text="Quante controllare (>=1):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_ant = ttk.Entry(estrazioni_frame_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_ant.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_ant.insert(0,"3")
buttons_frame_ant = ttk.Frame(controls_frame_ant); buttons_frame_ant.grid(row=0,column=2,sticky="ns",padx=(10,0))
button_cerca_ant = ttk.Button(buttons_frame_ant, text="Cerca Antecedenti",command=lambda:cerca_numeri(modalita="antecedenti")); button_cerca_ant.pack(pady=5,fill=tk.X,ipady=3)


common_controls_top_frame = ttk.Frame(main_frame); common_controls_top_frame.pack(fill=tk.X,pady=5)
dates_frame = ttk.LabelFrame(common_controls_top_frame, text=" Periodo Analisi (Comune) ",padding=5); dates_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y)
dates_frame.columnconfigure(1,weight=1)
ttk.Label(dates_frame,text="Da:",anchor="e").grid(row=0,column=0,padx=2,pady=2,sticky="w")
start_date_default = datetime.date.today()-datetime.timedelta(days=365*3)
start_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9),year=start_date_default.year,month=start_date_default.month,day=start_date_default.day)
start_date_entry.grid(row=0,column=1,padx=2,pady=2,sticky="ew")
ttk.Label(dates_frame,text="A:",anchor="e").grid(row=1,column=0,padx=2,pady=2,sticky="w")
end_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9))
end_date_entry.grid(row=1,column=1,padx=2,pady=2,sticky="ew")
common_buttons_frame = ttk.Frame(common_controls_top_frame); common_buttons_frame.pack(side=tk.LEFT,padx=10,fill=tk.Y)
button_salva = ttk.Button(common_buttons_frame,text="Salva Risultati",command=salva_risultati); button_salva.pack(side=tk.LEFT,pady=5,padx=5,ipady=3)
button_visualizza = ttk.Button(common_buttons_frame,text="Visualizza Grafici\n(Solo Successivi)",command=visualizza_grafici_successivi); button_visualizza.pack(side=tk.LEFT,pady=5,padx=5,ipady=0); button_visualizza.config(state=tk.DISABLED)


post_analysis_checks_frame = ttk.Frame(main_frame); post_analysis_checks_frame.pack(fill=tk.X,pady=(5,0))
verifica_futura_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Predittiva (Post-Analisi) ",padding=5); verifica_futura_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y,expand=False)
ttk.Label(verifica_futura_frame,text="Controlla N Colpi dopo Data Fine:",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_futura = ttk.Entry(verifica_futura_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_futura.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_futura.insert(0,"9")
button_verifica_futura = ttk.Button(verifica_futura_frame,text="Verifica Futura\n(Post-Analisi)",command=esegui_verifica_futura); button_verifica_futura.pack(pady=5,fill=tk.X,ipady=0); button_verifica_futura.config(state=tk.DISABLED)
verifica_mista_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Mista (su Trigger Spia) ",padding=5)
verifica_mista_frame.pack(side=tk.LEFT,padx=10,fill=tk.BOTH,expand=True) 
ttk.Label(verifica_mista_frame, text="Combinazioni (1-5 numeri, 1-90, una per riga, separate da '-' o spazio):", style="Small.TLabel").pack(anchor="w")
text_mista_container = ttk.Frame(verifica_mista_frame)
text_mista_container.pack(fill=tk.BOTH, expand=True, pady=(0,5)) 
text_combinazioni_miste_scrollbar_y = ttk.Scrollbar(text_mista_container, orient=tk.VERTICAL)
text_combinazioni_miste_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_combinazioni_miste = tk.Text(text_mista_container, height=4, width=30, 
                                  font=("Consolas", 10), wrap=tk.WORD,
                                  yscrollcommand=text_combinazioni_miste_scrollbar_y.set,
                                  bd=1, relief="sunken")
text_combinazioni_miste.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
text_combinazioni_miste_scrollbar_y.config(command=text_combinazioni_miste.yview)
text_combinazioni_miste.insert("1.0", "18 23 43\n60") 
ttk.Label(verifica_mista_frame, text="Controlla N Colpi Verifica (1-18):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_mista = ttk.Entry(verifica_mista_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10))
estrazioni_entry_verifica_mista.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_mista.insert(0,"9")
button_verifica_mista = ttk.Button(verifica_mista_frame,text="Verifica Mista\n(su Trigger Spia)",command=esegui_verifica_mista)
button_verifica_mista.pack(pady=5,fill=tk.X,ipady=0); button_verifica_mista.config(state=tk.DISABLED)

ttk.Label(main_frame, text="Risultati Analisi (Log):", style="Header.TLabel").pack(anchor="w",pady=(15,0)) 
risultato_outer_frame = ttk.Frame(main_frame); risultato_outer_frame.pack(fill=tk.BOTH,expand=True,pady=5)
risultato_scroll_y = ttk.Scrollbar(risultato_outer_frame,orient=tk.VERTICAL); risultato_scroll_y.pack(side=tk.RIGHT,fill=tk.Y)
risultato_scroll_x = ttk.Scrollbar(risultato_outer_frame,orient=tk.HORIZONTAL); risultato_scroll_x.pack(side=tk.BOTTOM,fill=tk.X)
risultato_text = tk.Text(risultato_outer_frame,wrap=tk.NONE,font=("Consolas",10),height=15,yscrollcommand=risultato_scroll_y.set,xscrollcommand=risultato_scroll_x.set,state=tk.DISABLED,bd=1,relief="sunken")
risultato_text.pack(fill=tk.BOTH,expand=True); risultato_scroll_y.config(command=risultato_text.yview); risultato_scroll_x.config(command=risultato_text.xview)

def aggiorna_lista_file_gui(target_listbox):
    global file_ruote
    target_listbox.config(state=tk.NORMAL); target_listbox.delete(0, tk.END)
    ruote_ordinate = sorted(file_ruote.keys())
    if ruote_ordinate: [target_listbox.insert(tk.END, r) for r in ruote_ordinate]
    else: target_listbox.insert(tk.END, "Nessun file ruota valido"); target_listbox.config(state=tk.DISABLED)

def mappa_file_ruote():
    global file_ruote, cartella_entry
    cartella = cartella_entry.get(); file_ruote.clear()
    if not cartella or not os.path.isdir(cartella): return False
    ruote_valide = ['BARI','CAGLIARI','FIRENZE','GENOVA','MILANO','NAPOLI','PALERMO','ROMA','TORINO','VENEZIA','NAZIONALE']
    found = False
    try:
        for file in os.listdir(cartella):
            fp = os.path.join(cartella, file)
            if os.path.isfile(fp) and file.lower().endswith(".txt"):
                nome_base = os.path.splitext(file)[0].upper()
                if nome_base in ruote_valide: file_ruote[nome_base] = fp; found = True
        return found
    except OSError as e: messagebox.showerror("Errore Lettura Cartella", f"Errore: {e}"); return False
    except Exception as e: messagebox.showerror("Errore Scansione", f"Errore: {e}"); traceback.print_exc(); return False

def on_sfoglia_click():
    global cartella_entry, listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant
    cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni")
    if cartella_sel:
        cartella_entry.delete(0,tk.END); cartella_entry.insert(0,cartella_sel)
        if mappa_file_ruote():
            aggiorna_lista_file_gui(listbox_ruote_analisi)
            aggiorna_lista_file_gui(listbox_ruote_verifica)
            aggiorna_lista_file_gui(listbox_ruote_analisi_ant)
        else:
            for lb in [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant]:
                lb.config(state=tk.NORMAL); lb.delete(0,tk.END); lb.insert(tk.END, "Nessun file valido"); lb.config(state=tk.DISABLED)
            messagebox.showwarning("Nessun File", "Nessun file .txt valido trovato.")
btn_sfoglia.config(command=on_sfoglia_click)

def main():
    global root, risultato_text
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "Benvenuto!\n\n1. Usa 'Sfoglia...' per cartella estrazioni.\n2. Seleziona modalità e parametri.\n   - Per 'Numeri Successivi', scegli Tipo Spia (Estratto/Ambo) e inserisci i numeri.\n3. Imposta periodo analisi.\n4. Clicca 'Cerca...'.\n5. Dopo 'Cerca Successivi', usa:\n   - Grafici.\n   - Verifica Futura (Post-Analisi): per i top combinati generati, solo dopo data fine analisi.\n   - Verifica Mista (su Trigger Spia): per combinazioni utente, basata sui trigger spia dell'analisi.\n\nIl riepilogo dell'analisi e delle verifiche apparirà anche in finestre popup separate.")
    risultato_text.config(state=tk.DISABLED)
    # on_sfoglia_click() # Rimosso per evitare apertura immediata
    root.mainloop()
    print("\nScript terminato.")

if __name__ == "__main__":
    main()