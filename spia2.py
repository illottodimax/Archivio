# -*- coding: utf-8 -*-
# Versione 3.8 - RITORNO AL FUNZIONANTE + FIX VARI (Incluso IndentationError)
# + AGGIUNTA VERIFICA MISTA

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import traceback # Per debug
import sys # Per controllo modalità interattiva
import itertools # Importato per le combinazioni
from collections import Counter # Importato per contare le posizioni dei colpi

# Prova a importare seaborn, ma continua anche se non è disponibile
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}

# Variabili GUI globali che saranno definite dopo
button_verifica_esiti = None
button_visualizza = None
button_verifica_futura = None
estrazioni_entry_verifica_futura = None
# Nuove variabili GUI per Verifica Mista
button_verifica_mista = None
entry_numeri_misti = []
estrazioni_entry_verifica_mista = None


# =============================================================================
# FUNZIONI GRAFICHE
# =============================================================================
def crea_grafico_barre(risultato, info_ricerca, tipo="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D']))
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        res_estratti = risultato.get('estratto', {})
        
        if tipo == "presenza":
            dati = res_estratti.get('presenza', {}).get('top', pd.Series(dtype='float64')).to_dict()
            percentuali = res_estratti.get('presenza', {}).get('percentuali', pd.Series(dtype='float64')).to_dict()
            base_conteggio = risultato.get('totale_trigger', 0)
            titolo = f"Presenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Serie Trigger ({base_conteggio} totali)"
        else: # tipo == "frequenza"
            dati = res_estratti.get('frequenza', {}).get('top', pd.Series(dtype='float64')).to_dict()
            percentuali = res_estratti.get('frequenza', {}).get('percentuali', pd.Series(dtype='float64')).to_dict()
            base_conteggio = sum(res_estratti.get('frequenza', {}).get('top', pd.Series(dtype='float64')).values()) if dati else 0
            titolo = f"Frequenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = "N. Occorrenze Totali (Estratti)"
            
        numeri = list(dati.keys())
        valori = list(dati.values())
        perc = [percentuali.get(num, 0.0) for num in numeri]
        
        if not numeri:
            ax.text(0.5, 0.5, "Nessun dato", ha='center', va='center')
            ax.set_title(titolo)
            ax.axis('off')
            plt.close(fig)
            return None
            
        bars = ax.bar(numeri, valori, color='skyblue', width=0.6)
        for i, (bar, p) in enumerate(zip(bars, perc)):
            h = bar.get_height()
            p_txt = f'{p:.1f}%' if p > 0.1 else '' # Mostra solo se perc > 0.1%
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, p_txt, ha='center', va='bottom', fontweight='bold', fontsize=9)
            
        ax.set_xlabel('Numeri Estratti su ' + ruota_verifica)
        ax.set_ylabel(ylabel)
        ax.set_title(titolo, fontsize=12)
        ax.set_ylim(0, max(valori or [1]) * 1.15) # Assicura che ci sia sempre un limite > 0
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {risultato.get('totale_trigger', 0)}"
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9)
        plt.tight_layout(pad=3.0)
        return fig
    except Exception as e:
        print(f"Errore in crea_grafico_barre: {e}")
        traceback.print_exc()
        if 'fig' in locals() and fig is not None:
            plt.close(fig)
        return None

def crea_tabella_lotto(risultato, info_ricerca, tipo="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D']))
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        n_trigger = risultato.get('totale_trigger', 0)
        numeri_lotto = np.arange(1, 91).reshape(9, 10)
        res_estratti = risultato.get('estratto', {})
        
        if tipo == "presenza":
            percentuali_serie = res_estratti.get('all_percentuali_presenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Presenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        else: # tipo == "frequenza"
            percentuali_serie = res_estratti.get('all_percentuali_frequenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Frequenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        
        percentuali = percentuali_serie.to_dict() if not percentuali_serie.empty else {}
        
        colors_norm = np.full(numeri_lotto.shape, 0.9) # Default light gray
        valid_perc = [p for p in percentuali.values() if pd.notna(p) and p > 0]
        max_perc = max(valid_perc) if valid_perc else 1.0 # Evita divisione per zero se non ci sono percentuali valide
        if max_perc == 0: max_perc = 1.0
        
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]
                num_str = str(num).zfill(2)
                num_str_alt = str(num) # Per chiavi non zfillate
                perc_val = percentuali.get(num_str)
                if perc_val is None: perc_val = percentuali.get(num_str_alt)
                
                if perc_val is not None and pd.notna(perc_val) and perc_val > 0:
                    colors_norm[i, j] = 0.9 - (0.9 * (perc_val / max_perc)) # Normalizza da 0 (max perc) a 0.9 (min perc)
        
        for r in range(10): # Griglia orizzontale e verticale
            ax.axvline(r - 0.5, color='gray', linestyle='-', alpha=0.3)
            ax.axhline(r - 0.5, color='gray', linestyle='-', alpha=0.3)
        
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]
                norm_color = colors_norm[i, j]
                cell_color = "white"
                text_color = "black"
                if norm_color < 0.9: # Solo se c'è una percentuale > 0
                    intensity = (0.9 - norm_color) / 0.9 # da 0 a 1
                    r_val = int(220 * (1 - intensity)) # da bianco a blu chiaro/scuro
                    g_val = int(230 * (1 - intensity))
                    b_val = int(255 * (1 - intensity / 2))
                    cell_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    if intensity > 0.6: text_color = "white" # Testo bianco su sfondi scuri
                
                edge_color = 'black' if norm_color < 0.9 else 'gray'
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cell_color, 
                                    alpha=1.0, edgecolor=edge_color, linewidth=1)
                ax.add_patch(rect)
                ax.text(j, i, num, ha="center", va="center", color=text_color, 
                       fontsize=10, fontweight="bold")
        
        ax.set_xlim(-0.5, 9.5); ax.set_ylim(8.5, -0.5) # Inverti asse y per tabella standard
        ax.set_xticks([]); ax.set_yticks([]) # Rimuovi ticks
        plt.title(titolo, fontsize=14, pad=15)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {n_trigger}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Lascia spazio per il testo in basso
        return fig
    
    except Exception as e:
        print(f"Errore in crea_tabella_lotto: {e}")
        traceback.print_exc()
        if 'fig' in locals() and fig is not None:
            plt.close(fig)
        return None

def crea_heatmap_correlazione(risultati, info_ricerca, tipo="presenza"):
    fig = None 
    try:
        if len(risultati) < 2: return None
        numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D']))
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        all_numeri_estratti = set(); percentuali_per_ruota = {}
        for ruota_v, _, res in risultati:
            if res is None or not isinstance(res, dict): continue
            res_tipo = res.get('estratto', {}); perc_dict = res_tipo.get(f'all_percentuali_{tipo}', pd.Series(dtype='float64')).to_dict()
            if not perc_dict: continue
            percentuali_per_ruota[ruota_v] = {num: perc for num, perc in perc_dict.items() if pd.notna(perc) and perc > 0}
            if percentuali_per_ruota[ruota_v]: all_numeri_estratti.update(percentuali_per_ruota[ruota_v].keys())
        if not percentuali_per_ruota or not all_numeri_estratti: return None
        all_numeri_sorted = sorted(list(all_numeri_estratti), key=int); ruote_heatmap = sorted(list(percentuali_per_ruota.keys()))
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
            cbar = fig.colorbar(im,ax=ax,shrink=0.7); cbar.ax.set_ylabel(f"% {tipo.capitalize()}",rotation=-90,va="bottom",fontsize=8); cbar.ax.tick_params(labelsize=7)
        plt.title(f"Heatmap {tipo.capitalize()} ESTRATTI\n(Spia {numeri_spia_str} su {ruote_analisi_str})",fontsize=11,pad=15); plt.xlabel("Numeri Estratti",fontsize=9); plt.ylabel("Ruote Verifica",fontsize=9)
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
        
        barre_frame = create_scrollable_tab(notebook, "Grafici a Barre (Estratti)")
        tabelle_frame = create_scrollable_tab(notebook, "Tabelle Lotto (Estratti)")
        heatmap_frame = create_scrollable_tab(notebook, "Heatmap Incrociata (Estratti)")

        for frame, label_text in [(barre_frame, "Grafici a Barre per Ruota di Verifica"), 
                                   (tabelle_frame, "Tabelle Lotto per Ruota di Verifica"), 
                                   (heatmap_frame, "Heatmap Incrociata tra Ruote di Verifica")]:
            ttk.Label(frame, text=label_text, style="Header.TLabel").pack(pady=10)

        for ruota_v, _, risultato in risultati_da_visualizzare:
            if risultato:
                info_specifica = info_globale_ricerca.copy(); info_specifica['ruota_verifica'] = ruota_v
                # Barre
                ruota_bar_frame = ttk.LabelFrame(barre_frame, text=f"Ruota Verifica: {ruota_v}"); ruota_bar_frame.pack(fill="x",expand=False,padx=10,pady=10)
                fig_p = crea_grafico_barre(risultato,info_specifica,"presenza"); fig_f = crea_grafico_barre(risultato,info_specifica,"frequenza")
                if fig_p: FigureCanvasTkAgg(fig_p, master=ruota_bar_frame).get_tk_widget().pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                if fig_f: FigureCanvasTkAgg(fig_f, master=ruota_bar_frame).get_tk_widget().pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
                if not fig_p and not fig_f: ttk.Label(ruota_bar_frame,text="Nessun grafico generato").pack(padx=5,pady=5)
                # Tabelle
                ruota_tab_frame = ttk.LabelFrame(tabelle_frame, text=f"Ruota Verifica: {ruota_v}"); ruota_tab_frame.pack(fill="x",expand=False,padx=10,pady=10)
                fig_tp = crea_tabella_lotto(risultato,info_specifica,"presenza"); fig_tf = crea_tabella_lotto(risultato,info_specifica,"frequenza")
                if fig_tp: FigureCanvasTkAgg(fig_tp, master=ruota_tab_frame).get_tk_widget().pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                if fig_tf: FigureCanvasTkAgg(fig_tf, master=ruota_tab_frame).get_tk_widget().pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
                if not fig_tp and not fig_tf: ttk.Label(ruota_tab_frame,text="Nessuna tabella generata").pack(padx=5,pady=5)

        risultati_validi_heatmap = [r for r in risultati_da_visualizzare if r[2] and isinstance(r[2],dict)]
        if len(risultati_validi_heatmap) < 2: ttk.Label(heatmap_frame,text="Richiede >= 2 Ruote Verifica con risultati validi.").pack(padx=20,pady=20)
        else:
            for tipo_h, nome_h in [("presenza","Presenza"),("frequenza","Frequenza")]:
                heatmap_fig = crea_heatmap_correlazione(risultati_validi_heatmap,info_globale_ricerca,tipo_h)
                if heatmap_fig: 
                    ttk.Label(heatmap_frame,text=f"--- Heatmap {nome_h} ---",font=("Helvetica",11,"bold")).pack(pady=(10,5))
                    FigureCanvasTkAgg(heatmap_fig,master=heatmap_frame).get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)
                else: ttk.Label(heatmap_frame,text=f"Nessuna Heatmap {nome_h} generata.").pack(pady=10)
    except Exception as e: messagebox.showerror("Errore Visualizzazione",f"Errore creazione finestra grafici:\n{e}"); traceback.print_exc()
    finally: plt.close('all')

# =============================================================================
# FUNZIONI DI LOGICA
# (OMESSE PER BREVITÀ - SONO UGUALI ALLA VERSIONE 3.8.12)
# =============================================================================
# ... (carica_dati, analizza_ruota_verifica, analizza_antecedenti, 
#      aggiorna_risultati_globali, salva_risultati, format_ambo_terno, cerca_numeri,
#      verifica_esiti_combinati, esegui_verifica_esiti,
#      verifica_esiti_utente_su_triggers, 
#      verifica_esiti_futuri, esegui_verifica_futura, esegui_verifica_mista,
#      visualizza_grafici_successivi)
#      sono uguali alla versione 3.8.12 e sono state già fornite.
#      Per mantenere questa risposta focalizzata sul codice completo e pulito,
#      le riporterò di seguito, assicurandomi che siano le versioni corrette e pulite.

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
                numeri_estratti = sorted([row[col] for col in colonne_numeri if pd.notna(row[col])])
                if not numeri_estratti: continue
                for num in numeri_estratti: freq_estratti[num] = freq_estratti.get(num, 0) + 1; estratti_unici_finestra.add(num)
                if len(numeri_estratti) >= 2:
                    for ambo in itertools.combinations(numeri_estratti, 2): freq_ambi[ambo] = freq_ambi.get(ambo, 0) + 1; ambi_unici_finestra.add(ambo)
                if len(numeri_estratti) >= 3:
                    for terno in itertools.combinations(numeri_estratti, 3): freq_terne[terno] = freq_terne.get(terno, 0) + 1; terne_unici_finestra.add(terno)
        for num in estratti_unici_finestra: pres_estratti[num] = pres_estratti.get(num, 0) + 1
        for ambo in ambi_unici_finestra: pres_ambi[ambo] = pres_ambi.get(ambo, 0) + 1
        for terno in terne_unici_finestra: pres_terne[terno] = pres_terne.get(terno, 0) + 1
    results = {'totale_trigger': n_trigger}
    for tipo, freq_dict, pres_dict in [('estratto', freq_estratti, pres_estratti), ('ambo', freq_ambi, pres_ambi), ('terno', freq_terne, pres_terne)]:
        if not freq_dict: results[tipo] = None; continue
        freq_s = pd.Series(freq_dict, dtype=int).sort_index(); pres_s = pd.Series(pres_dict, dtype=int).reindex(freq_s.index, fill_value=0).sort_index()
        tot_freq = freq_s.sum(); perc_freq = (freq_s/tot_freq*100).round(2) if tot_freq > 0 else pd.Series(0.0, index=freq_s.index)
        perc_pres = (pres_s/n_trigger*100).round(2) if n_trigger > 0 else pd.Series(0.0, index=pres_s.index)
        top_pres = pres_s.sort_values(ascending=False).head(10); top_freq = freq_s.sort_values(ascending=False).head(10)
        results[tipo] = {'presenza': {'top':top_pres, 'percentuali':perc_pres.reindex(top_pres.index).fillna(0.0), 'frequenze':freq_s.reindex(top_pres.index).fillna(0).astype(int), 'perc_frequenza':perc_freq.reindex(top_pres.index).fillna(0.0)},
                         'frequenza':{'top':top_freq, 'percentuali':perc_freq.reindex(top_freq.index).fillna(0.0), 'presenze':pres_s.reindex(top_freq.index).fillna(0).astype(int), 'perc_presenza':perc_pres.reindex(top_freq.index).fillna(0.0)},
                         'all_percentuali_presenza':perc_pres, 'all_percentuali_frequenza':perc_freq, 'full_presenze':pres_s, 'full_frequenze':freq_s}
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
    global button_verifica_esiti, button_visualizza, button_verifica_futura, button_verifica_mista
    if button_verifica_esiti: button_verifica_esiti.config(state=tk.DISABLED)
    if button_visualizza: button_visualizza.config(state=tk.DISABLED)
    if button_verifica_futura: button_verifica_futura.config(state=tk.DISABLED)
    if button_verifica_mista: button_verifica_mista.config(state=tk.DISABLED)
    if modalita == "successivi":
        risultati_globali = risultati_nuovi if risultati_nuovi is not None else []
        info_ricerca_globale = info_ricerca if info_ricerca is not None else {}
        has_valid_results = bool(risultati_globali) and any(res[2] for res in risultati_globali if len(res)>2)
        has_top_combinati = bool(info_ricerca_globale.get('top_combinati')) and any(info_ricerca_globale['top_combinati'].values())
        has_date_trigger = bool(info_ricerca_globale.get('date_trigger_ordinate'))
        has_end_date = info_ricerca_globale.get('end_date') is not None
        has_ruote_verifica_info = bool(info_ricerca_globale.get('ruote_verifica'))
        if has_valid_results and button_visualizza: button_visualizza.config(state=tk.NORMAL)
        if has_valid_results and has_top_combinati and has_date_trigger and button_verifica_esiti: button_verifica_esiti.config(state=tk.NORMAL)
        if has_end_date and has_ruote_verifica_info and button_verifica_futura:
             button_verifica_futura.config(state=tk.NORMAL)
        if has_date_trigger and has_ruote_verifica_info and button_verifica_mista: 
            button_verifica_mista.config(state=tk.NORMAL)
    else: risultati_globali, info_ricerca_globale = [], {}

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

def format_ambo_terno(combinazione): return "-".join(map(str, combinazione))

def cerca_numeri(modalita="successivi"):
    global risultati_globali, info_ricerca_globale, file_ruote, risultato_text, root
    global start_date_entry, end_date_entry, listbox_ruote_analisi, listbox_ruote_verifica
    global entry_numeri_spia, estrazioni_entry_succ, listbox_ruote_analisi_ant
    global entry_numeri_obiettivo, estrazioni_entry_ant
    if not mappa_file_ruote() or not file_ruote: messagebox.showerror("Errore File","Mappa file ruote fallita."); return
    risultati_globali,info_ricerca_globale = [],{}; risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,f"Ricerca {modalita}...\n");risultato_text.see(tk.END);root.update_idletasks()
    aggiorna_risultati_globali([],{},modalita=modalita)
    try:
        start_dt,end_dt = start_date_entry.get_date(),end_date_entry.get_date()
        if start_dt > end_dt: raise ValueError("Data inizio dopo data fine.")
        start_ts,end_ts = pd.Timestamp(start_dt),pd.Timestamp(end_dt)
    except Exception as e: messagebox.showerror("Input Date",f"Date non valide: {e}"); return
    messaggi_out,ris_graf_loc = [],[]; col_num = [f'Numero{i+1}' for i in range(5)]
    if modalita == "successivi":
        ra_idx,rv_idx = listbox_ruote_analisi.curselection(),listbox_ruote_verifica.curselection()
        if not ra_idx or not rv_idx: messagebox.showwarning("Manca Input","Seleziona Ruote An/Ver."); return
        nomi_ra,nomi_rv = [listbox_ruote_analisi.get(i) for i in ra_idx],[listbox_ruote_verifica.get(i) for i in rv_idx]
        num_spia = sorted(list(set(str(int(e.get().strip())).zfill(2) for e in entry_numeri_spia if e.get().strip() and e.get().strip().isdigit() and 1<=int(e.get().strip())<=90)))
        if not num_spia: messagebox.showwarning("Manca Input","Numeri Spia non validi."); return
        try: n_estr=int(estrazioni_entry_succ.get()); assert 1<=n_estr<=18
        except: messagebox.showerror("Input Invalido","N. Estrazioni (1-18) non valido."); return
        info_curr={'numeri_spia':num_spia,'ruote_analisi':nomi_ra,'ruote_verifica':nomi_rv,'n_estrazioni':n_estr,'start_date':start_ts,'end_date':end_ts}
        all_date_trig=set(); messaggi_out.append("--- FASE 1: Ricerca Date Uscita Spia ---")
        for nome_ra in nomi_ra:
            df_an=carica_dati(file_ruote.get(nome_ra),start_ts,end_ts)
            if df_an is None or df_an.empty: messaggi_out.append(f"[{nome_ra}] No dati An."); continue
            dates_found_arr=df_an.loc[df_an[col_num].isin(num_spia).any(axis=1),'Data'].unique()
            dates_found=pd.to_datetime(dates_found_arr)
            if dates_found.size > 0: all_date_trig.update(dates_found); messaggi_out.append(f"[{nome_ra}] Trovate {len(dates_found)} date trigger.")
            else: messaggi_out.append(f"[{nome_ra}] Nessuna uscita spia.")
        if not all_date_trig: messaggi_out.append(f"\nNESSUNA USCITA SPIA TROVATA."); aggiorna_risultati_globali([],{},modalita=modalita)
        else:
            date_trig_ord=sorted(list(all_date_trig)); n_trig_tot=len(date_trig_ord)
            messaggi_out.append(f"\nFASE 1 OK: {n_trig_tot} date trigger totali."); info_curr['date_trigger_ordinate']=date_trig_ord
            messaggi_out.append("\n--- FASE 2: Analisi Ruote Verifica ---"); df_cache_ver={}; num_rv_ok=0
            for nome_rv in nomi_rv:
                df_ver_full = df_cache_ver.get(nome_rv)
                if df_ver_full is None: df_ver_full=carica_dati(file_ruote.get(nome_rv),start_ts,end_ts); df_cache_ver[nome_rv]=df_ver_full
                if df_ver_full is None or df_ver_full.empty: messaggi_out.append(f"[{nome_rv}] No dati Ver."); continue
                res_ver,err_ver=analizza_ruota_verifica(df_ver_full,date_trig_ord,n_estr,nome_rv)
                if err_ver: messaggi_out.append(f"[{nome_rv}] Errore: {err_ver}"); continue
                if res_ver:
                    ris_graf_loc.append((nome_rv,", ".join(num_spia),res_ver)); num_rv_ok+=1
                    msg_res_v=f"\n=== Risultati Verifica: {nome_rv} (Base: {res_ver['totale_trigger']} trigger) ==="
                    for tipo_s in ['estratto','ambo','terno']:
                        res_s=res_ver.get(tipo_s)
                        if res_s:
                            msg_res_v+=f"\n--- {tipo_s.capitalize()} ---\n  Top Presenza:\n"
                            if not res_s['presenza']['top'].empty:
                                for i,(item,pres) in enumerate(res_s['presenza']['top'].items()):
                                    item_str=format_ambo_terno(item) if isinstance(item,tuple) else item
                                    perc_p=res_s['presenza']['percentuali'].get(item,0.0); freq_p=res_s['presenza']['frequenze'].get(item,0)
                                    msg_res_v+=f"    {i+1}. {item_str}: {pres} ({perc_p:.1f}%) [F:{freq_p}]\n"
                            else: msg_res_v+="    Nessuno.\n"
                        else: msg_res_v+=f"\n--- {tipo_s.capitalize()}: Nessun risultato ---\n"
                    messaggi_out.append(msg_res_v)
                messaggi_out.append("- "*20)
            if ris_graf_loc and num_rv_ok > 0:
                messaggi_out.append("\n\n=== RISULTATI COMBINATI (Tutte Ruote Verifica) ===")
                top_comb_ver={'estratto':[],'ambo':[],'terno':[]}; peso_pres,peso_freq=0.6,0.4
                for tipo in ['estratto','ambo','terno']:
                    messaggi_out.append(f"\n--- Combinati: {tipo.upper()} ---"); comb_pres_dict,comb_freq_dict,has_data={},{},False
                    for _,_,res in ris_graf_loc:
                        if res and res.get(tipo):
                            has_data=True
                            for item,count in res[tipo].get('full_presenze',pd.Series(dtype=int)).items(): comb_pres_dict[item]=comb_pres_dict.get(item,0)+count
                            for item,count in res[tipo].get('full_frequenze',pd.Series(dtype=int)).items(): comb_freq_dict[item]=comb_freq_dict.get(item,0)+count
                    if not has_data: messaggi_out.append(f"    Nessun risultato combinato per {tipo}.\n"); continue
                    comb_pres,comb_freq=pd.Series(comb_pres_dict,dtype=int),pd.Series(comb_freq_dict,dtype=int)
                    all_items_idx=comb_pres.index.union(comb_freq.index); comb_pres=comb_pres.reindex(all_items_idx,fill_value=0).sort_index(); comb_freq=comb_freq.reindex(all_items_idx,fill_value=0).sort_index()
                    tot_pres_ops=n_trig_tot*num_rv_ok; comb_perc_pres=(comb_pres/tot_pres_ops*100) if tot_pres_ops>0 else pd.Series(0.0,index=comb_pres.index)
                    max_freq=comb_freq.max(); comb_freq_norm=(comb_freq/max_freq*100) if max_freq>0 else pd.Series(0.0,index=comb_freq.index)
                    punt_comb=((peso_pres*comb_perc_pres)+(peso_freq*comb_freq_norm)).round(2).sort_values(ascending=False)
                    top_punt=punt_comb.head(10)
                    if not top_punt.empty: top_comb_ver[tipo]=top_punt.head(5).index.tolist()
                    messaggi_out.append(f"  Top 10 Combinati per Punteggio:\n")
                    if not top_punt.empty:
                        for i,(item,score) in enumerate(top_punt.items()):
                            item_str=format_ambo_terno(item) if isinstance(item,tuple) else item
                            messaggi_out.append(f"    {i+1}. {item_str}: Punt={score:.2f} (PresAvg:{comb_perc_pres.get(item,0.0):.1f}%, FreqTot:{comb_freq.get(item,0)})\n")
                    else: messaggi_out.append("    Nessuno.\n")
                info_curr['top_combinati']=top_comb_ver
            elif num_rv_ok == 0: messaggi_out.append("\nNessuna Ruota Verifica valida con risultati.")
        aggiorna_risultati_globali(ris_graf_loc,info_curr,modalita="successivi")
    elif modalita == "antecedenti":
        ra_ant_idx=listbox_ruote_analisi_ant.curselection()
        if not ra_ant_idx: messagebox.showwarning("Manca Input","Seleziona Ruota/e Analisi."); return
        nomi_ra_ant=[listbox_ruote_analisi_ant.get(i) for i in ra_ant_idx]
        num_obj=sorted(list(set(str(int(e.get().strip())).zfill(2) for e in entry_numeri_obiettivo if e.get().strip() and e.get().strip().isdigit() and 1<=int(e.get().strip())<=90)))
        if not num_obj: messagebox.showwarning("Manca Input","Numeri Obiettivo non validi."); return
        try: n_prec=int(estrazioni_entry_ant.get()); assert n_prec >=1
        except: messagebox.showerror("Input Invalido","N. Precedenti (>=1) non valido."); return
        messaggi_out.append(f"--- Analisi Antecedenti (Marker) ---"); df_cache_ant={}
        for nome_ra_ant in nomi_ra_ant:
            df_ant_full = df_cache_ant.get(nome_ra_ant)
            if df_ant_full is None: df_ant_full=carica_dati(file_ruote.get(nome_ra_ant),start_ts,end_ts); df_cache_ant[nome_ra_ant]=df_ant_full
            if df_ant_full is None or df_ant_full.empty: messaggi_out.append(f"[{nome_ra_ant}] No dati."); continue
            df_ruota_curr=df_ant_full
            res_ant,err_ant=analizza_antecedenti(df_ruota_curr,num_obj,n_prec,nome_ra_ant)
            if err_ant: messaggi_out.append(f"[{nome_ra_ant}] Errore: {err_ant}"); continue
            if res_ant and res_ant.get('base_presenza_antecedenti',0)>0 and (not res_ant['presenza']['top'].empty or not res_ant['frequenza']['top'].empty):
                msg_res_ant=f"\n=== Risultati Antecedenti: {nome_ra_ant} ===\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Prec: {res_ant['n_precedenti']} | Occ.Ob.: {res_ant['totale_occorrenze_obiettivo']})"
                msg_res_ant+=f"\n  Top Antecedenti per Presenza:\n"
                if not res_ant['presenza']['top'].empty:
                    for i,(num,pres) in enumerate(res_ant['presenza']['top'].head(10).items()): msg_res_ant+=f"    {i+1}. {num}: {pres} ({res_ant['presenza']['percentuali'].get(num,0.0):.1f}%) [F:{res_ant['presenza']['frequenze'].get(num,0)}]\n"
                else: msg_res_ant+="    Nessuno.\n"
                messaggi_out.append(msg_res_ant)
            else: messaggi_out.append(f"[{nome_ra_ant}] Nessun dato antecedente significativo.")
            messaggi_out.append("- "*20)
        aggiorna_risultati_globali([],{},modalita="antecedenti")
    final_output="\n".join(messaggi_out) if messaggi_out else "Nessun risultato."
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,final_output); risultato_text.config(state=tk.DISABLED); risultato_text.see("1.0")

# =============================================================================
# FUNZIONI PER VERIFICA ESITI
# =============================================================================
# (verifica_esiti_combinati, esegui_verifica_esiti, verifica_esiti_futuri, esegui_verifica_futura, esegui_verifica_mista
#  sono uguali alla versione 3.8.11, ma verifica_esiti_utente_su_triggers è stata inserita sopra ed è quella utilizzata da esegui_verifica_mista)
#  Riporto verifica_esiti_utente_su_triggers (pulita) e le altre.

def verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_ruote_verifica, n_verifiche, start_ts, end_ts, titolo_sezione="VERIFICA MISTA SU TRIGGER"):
    if not date_triggers or not combinazioni_utente or not nomi_ruote_verifica: 
        return "Errore: Dati input mancanti per verifica utente su triggers."
    
    estratti_u = sorted(list(set(combinazioni_utente.get('estratto', []))))
    ambi_u_tpl = sorted(list(set(tuple(sorted(a)) for a in combinazioni_utente.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_u_tpl = sorted(list(set(tuple(sorted(t)) for t in combinazioni_utente.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
    quaterne_u_tpl = sorted(list(set(tuple(sorted(q)) for q in combinazioni_utente.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_u_tpl = sorted(list(set(tuple(sorted(c)) for c in combinazioni_utente.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))

    hit_details = {
        'estratto': {e: {'h':0,'p':[]} for e in estratti_u}, 
        'ambo': {a: {'h':0,'p':[]} for a in ambi_u_tpl}, 
        'terno': {t: {'h':0,'p':[]} for t in terni_u_tpl},
        'quaterna': {q: {'h':0,'p':[]} for q in quaterne_u_tpl},
        'cinquina': {c: {'h':0,'p':[]} for c in cinquine_u_tpl}
    }
    
    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver = {}; ruote_valide = []
    for nome_rv_loop in nomi_ruote_verifica:
        df_ver = carica_dati(file_ruote.get(nome_rv_loop), start_date=start_ts, end_date=None) # Carica dati estesi
        if df_ver is not None and not df_ver.empty: 
            df_cache_ver[nome_rv_loop] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
            ruote_valide.append(nome_rv_loop)
            
    if not ruote_valide: return "Errore: Nessuna ruota di verifica valida per caricare i dati del periodo."
    
    casi_tot_eff = len(date_triggers) * len(ruote_valide)
    if casi_tot_eff == 0: return "Nessun caso trigger da verificare."

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
                found_in_this_trigger_window = {'estratto':set(), 'ambo':set(), 'terno':set(), 'quaterna':set(), 'cinquina':set()}
                for colpo_idx, (_, row) in enumerate(df_fin_v.iterrows(), 1):
                    current_row_numbers_v = [row[col] for col in cols_num if pd.notna(row[col])]
                    nums_draw = sorted(current_row_numbers_v) 
                    if not nums_draw: continue 
                    set_nums_draw = set(nums_draw)
                    
                    if estratti_u:
                        for num_hit in set_nums_draw.intersection(set(estratti_u)):
                            if num_hit not in found_in_this_trigger_window['estratto']: 
                                hit_details['estratto'][num_hit]['h']+=1; hit_details['estratto'][num_hit]['p'].append(colpo_idx); found_in_this_trigger_window['estratto'].add(num_hit)
                    if ambi_u_tpl and len(nums_draw) >= 2:
                        ambi_generati_da_riga = set(itertools.combinations(nums_draw, 2))
                        for ambo_hit in ambi_generati_da_riga.intersection(set(ambi_u_tpl)):
                            if ambo_hit not in found_in_this_trigger_window['ambo']: 
                                hit_details['ambo'][ambo_hit]['h']+=1; hit_details['ambo'][ambo_hit]['p'].append(colpo_idx); found_in_this_trigger_window['ambo'].add(ambo_hit)
                    if terni_u_tpl and len(nums_draw) >= 3:
                        terni_generati_da_riga = set(itertools.combinations(nums_draw, 3))
                        for terno_hit in terni_generati_da_riga.intersection(set(terni_u_tpl)): 
                            if terno_hit not in found_in_this_trigger_window['terno']: 
                                hit_details['terno'][terno_hit]['h']+=1; hit_details['terno'][terno_hit]['p'].append(colpo_idx); found_in_this_trigger_window['terno'].add(terno_hit)
                    if quaterne_u_tpl and len(nums_draw) >= 4:
                        quaterne_generati_da_riga = set(itertools.combinations(nums_draw, 4))
                        for quaterna_hit in quaterne_generati_da_riga.intersection(set(quaterne_u_tpl)):
                            if quaterna_hit not in found_in_this_trigger_window['quaterna']: 
                                hit_details['quaterna'][quaterna_hit]['h']+=1; hit_details['quaterna'][quaterna_hit]['p'].append(colpo_idx); found_in_this_trigger_window['quaterna'].add(quaterna_hit)
                    if cinquine_u_tpl and len(nums_draw) >= 5:
                        cinquine_generati_da_riga = set(itertools.combinations(nums_draw, 5))
                        for cinquina_hit in cinquine_generati_da_riga.intersection(set(cinquine_u_tpl)):
                            if cinquina_hit not in found_in_this_trigger_window['cinquina']: 
                                hit_details['cinquina'][cinquina_hit]['h']+=1; hit_details['cinquina'][cinquina_hit]['p'].append(colpo_idx); found_in_this_trigger_window['cinquina'].add(cinquina_hit)
    
    out = [f"\n\n=== {titolo_sezione} ({n_verifiche} Colpi dopo ogni Trigger) ==="]
    out.append(f"Numero di casi trigger totali considerati (Date Trigger * Ruote Verifica): {casi_tot_eff}")
    sorti_verificate = [
        ('estratto', estratti_u), ('ambo', ambi_u_tpl), ('terno', terni_u_tpl),
        ('quaterna', quaterne_u_tpl), ('cinquina', cinquine_u_tpl)
    ]
    for tipo, items_v in sorti_verificate:
        if not items_v: continue
        out.append(f"\n--- Esiti {tipo.upper()} ---")
        for item in items_v:
            item_str = format_ambo_terno(item) if isinstance(item, tuple) else item
            details = hit_details[tipo].get(item)
            if not details: out.append(f"    - {item_str}: Dettagli non trovati."); continue
            n_vinc = details['h']; perc_vinc = round(n_vinc/casi_tot_eff*100,2) if casi_tot_eff > 0 else 0.0
            perc_sfald = 100.0 - perc_vinc
            pos_colpi = details['p']; pos_sum = "Nessuna vincita" if not pos_colpi else ", ".join([f"C{p}:{c}v" for p,c in sorted(Counter(pos_colpi).items())])
            avg_pos_str = f"{round(sum(pos_colpi)/len(pos_colpi),1)}" if pos_colpi else "N/A"
            out.append(f"    - {item_str}: {n_vinc} vincite su {casi_tot_eff} casi ({perc_vinc}%)")
            out.append(f"      └─ Sfaldamento: {perc_sfald:.1f}% | Uscite per Colpo: {pos_sum} (Media Colpo Uscita: {avg_pos_str})")
    return "\n".join(out)

# Funzione per la "Verifica Esiti Classica" (sui top_combinati dell'analisi)
def verifica_esiti_combinati(date_triggers, top_combinati, nomi_ruote_verifica, n_verifiche, start_ts, end_ts):
    # Questa funzione è ora un wrapper per verifica_esiti_utente_su_triggers
    # ma prepara il dizionario combinazioni_utente dai top_combinati
    combinazioni_da_top = {
        'estratto': top_combinati.get('estratto', []),
        'ambo': top_combinati.get('ambo', []),
        'terno': top_combinati.get('terno', [])
        # La verifica classica di solito non va oltre il terno per i top_combinati
    }
    return verifica_esiti_utente_su_triggers(
        date_triggers, combinazioni_da_top, nomi_ruote_verifica, 
        n_verifiche, start_ts, end_ts, 
        titolo_sezione="VERIFICA ESITI CLASSICA (Top Combinati)"
    )

def esegui_verifica_esiti():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica
    risultato_text.config(state=tk.NORMAL)
    risultato_text.insert(tk.END, "\n\nVerifica esiti classica (Top Combinati)...")
    risultato_text.see(tk.END); root.update_idletasks()
    date_triggers = info_ricerca_globale.get('date_trigger_ordinate')
    top_combinati = info_ricerca_globale.get('top_combinati')
    nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date')
    end_ts = info_ricerca_globale.get('end_date')
    if not all([date_triggers, top_combinati, nomi_rv, start_ts, end_ts]):
        messagebox.showerror("Errore Verifica Esiti", "Dati analisi 'Successivi' mancanti."); 
        risultato_text.config(state=tk.DISABLED); return
    if not any(top_combinati.values()) or sum(len(v) for v in top_combinati.values() if v) == 0:
        messagebox.showinfo("Verifica Esiti", "Nessun 'Top Combinato' da verificare."); 
        risultato_text.config(state=tk.DISABLED); return
    try: 
        n_ver = int(estrazioni_entry_verifica.get()); assert 1 <= n_ver <= 18
    except: 
        messagebox.showerror("Input Invalido", "N. Estrazioni Verifica (1-18) non valido."); 
        risultato_text.config(state=tk.DISABLED); return
    try: 
        res_str = verifica_esiti_combinati(date_triggers, top_combinati, nomi_rv, n_ver, start_ts, end_ts)
        risultato_text.insert(tk.END, res_str)
    except Exception as e: 
        risultato_text.insert(tk.END, f"\nErrore verifica esiti classica: {e}"); traceback.print_exc()
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)

def verifica_esiti_futuri(top_combinati_input, nomi_ruote_verifica, data_fine_analisi, n_colpi_futuri):
    if not top_combinati_input or not any(top_combinati_input.values()) or not nomi_ruote_verifica or data_fine_analisi is None or n_colpi_futuri <= 0:
        return "Errore: Input invalidi per verifica_esiti_futuri (post-analisi)."
    estratti_items = sorted(list(set(top_combinati_input.get('estratto', []))))
    ambi_items = sorted(list(set(tuple(sorted(a)) for a in top_combinati_input.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_items = sorted(list(set(tuple(sorted(t)) for t in top_combinati_input.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
    quaterne_items = sorted(list(set(tuple(sorted(q)) for q in top_combinati_input.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_items = sorted(list(set(tuple(sorted(c)) for c in top_combinati_input.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))
    set_estratti = set(estratti_items); set_ambi = set(ambi_items); set_terni = set(terni_items)
    set_quaterne = set(quaterne_items); set_cinquine = set(cinquine_items)
    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver_fut = {}; ruote_con_dati_fut = []
    for nome_rv in nomi_ruote_verifica:
        df_ver_full = carica_dati(file_ruote.get(nome_rv), start_date=None, end_date=None)
        if df_ver_full is None or df_ver_full.empty: continue
        df_ver_fut = df_ver_full[df_ver_full['Data'] > data_fine_analisi].copy().sort_values(by='Data').reset_index(drop=True)
        df_fin_fut = df_ver_fut.head(n_colpi_futuri)
        if not df_fin_fut.empty: df_cache_ver_fut[nome_rv] = df_fin_fut; ruote_con_dati_fut.append(nome_rv)
    if not ruote_con_dati_fut: return f"Nessuna estrazione trovata su nessuna ruota di verifica dopo {data_fine_analisi.date()} per {n_colpi_futuri} colpi."
    hits_registrati = {
        'estratto': {e: [] for e in estratti_items}, 'ambo': {a: [] for a in ambi_items},
        'terno': {t: [] for t in terni_items}, 'quaterna': {q: [] for q in quaterne_items},
        'cinquina': {c: [] for c in cinquine_items}
    }
    primo_hit_assoluto = {'estratto': set(), 'ambo': set(), 'terno': set(), 'quaterna': set(), 'cinquina': set()}
    for nome_rv in ruote_con_dati_fut:
        df_finestra_ruota = df_cache_ver_fut[nome_rv]
        for colpo_idx, (_, row) in enumerate(df_finestra_ruota.iterrows(), 1):
            numeri_estratti_riga = [row[col] for col in cols_num if pd.notna(row[col])]
            numeri_sortati_riga = sorted(numeri_estratti_riga)
            if not numeri_sortati_riga: continue 
            set_numeri_riga = set(numeri_sortati_riga)
            if estratti_items:
                for item_e in set_numeri_riga.intersection(set_estratti):
                    if ('estratto', item_e) not in primo_hit_assoluto['estratto']:
                        hits_registrati['estratto'][item_e].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['estratto'].add(('estratto', item_e))
            if ambi_items and len(numeri_sortati_riga) >= 2:
                for item_a in set(itertools.combinations(numeri_sortati_riga,2)).intersection(set_ambi):
                    if ('ambo', item_a) not in primo_hit_assoluto['ambo']:
                        hits_registrati['ambo'][item_a].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['ambo'].add(('ambo', item_a))
            if terni_items and len(numeri_sortati_riga) >= 3:
                for item_t in set(itertools.combinations(numeri_sortati_riga,3)).intersection(set_terni):
                    if ('terno', item_t) not in primo_hit_assoluto['terno']:
                        hits_registrati['terno'][item_t].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['terno'].add(('terno', item_t))
            if quaterne_items and len(numeri_sortati_riga) >= 4:
                for item_q in set(itertools.combinations(numeri_sortati_riga,4)).intersection(set_quaterne):
                    if ('quaterna', item_q) not in primo_hit_assoluto['quaterna']:
                        hits_registrati['quaterna'][item_q].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['quaterna'].add(('quaterna', item_q))
            if cinquine_items and len(numeri_sortati_riga) >= 5:
                for item_c in set(itertools.combinations(numeri_sortati_riga,5)).intersection(set_cinquine):
                    if ('cinquina', item_c) not in primo_hit_assoluto['cinquina']:
                        hits_registrati['cinquina'][item_c].append((nome_rv, colpo_idx, row['Data'].date())); primo_hit_assoluto['cinquina'].add(('cinquina', item_c))
    out = [f"\n\n=== VERIFICA ESITI FUTURI (POST-ANALISI) ({n_colpi_futuri} Colpi dopo {data_fine_analisi.date()}) ==="]
    out.append(f"Ruote verificate con dati futuri disponibili: {', '.join(ruote_con_dati_fut) or 'Nessuna'}")
    sorti_config = [
        ('estratto', estratti_items), ('ambo', ambi_items), ('terno', terni_items),
        ('quaterna', quaterne_items), ('cinquina', cinquine_items)
    ]
    for tipo_sorte, lista_items_sorte in sorti_config:
        if not lista_items_sorte: continue 
        out.append(f"\n--- Esiti Futuri {tipo_sorte.upper()} ---")
        almeno_un_hit_per_sorte = False
        for item_da_verificare in lista_items_sorte:
            item_str_formattato = format_ambo_terno(item_da_verificare) if isinstance(item_da_verificare, tuple) else item_da_verificare
            dettagli_hit_item = hits_registrati[tipo_sorte].get(item_da_verificare, [])
            if dettagli_hit_item:
                almeno_un_hit_per_sorte = True
                dettagli_ordinati = sorted(dettagli_hit_item, key=lambda x: (x[1], x[0]))
                stringa_dettagli = "; ".join([f"{d_ruota} @ C{d_colpo} ({d_data})" for d_ruota, d_colpo, d_data in dettagli_ordinati])
                out.append(f"    - {item_str_formattato}: USCITO -> {stringa_dettagli}")
            else:
                out.append(f"    - {item_str_formattato}: NON uscito")
        if not almeno_un_hit_per_sorte and lista_items_sorte:
             out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito nei colpi futuri analizzati.")
    return "\n".join(out)

def esegui_verifica_futura():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica_futura
    risultato_text.config(state=tk.NORMAL); risultato_text.insert(tk.END, "\n\nVerifica esiti futuri (post-analisi)..."); risultato_text.see(tk.END); root.update_idletasks()
    top_c = info_ricerca_globale.get('top_combinati'); nomi_rv = info_ricerca_globale.get('ruote_verifica')
    data_fine = info_ricerca_globale.get('end_date')
    if not all([top_c, nomi_rv, data_fine]): 
        messagebox.showerror("Errore Verifica Futura", "Dati analisi 'Successivi' (Top combinati, Ruote verifica, Data Fine) mancanti."); 
        risultato_text.config(state=tk.DISABLED); return
    if not any(v for v in top_c.values() if isinstance(v, list) and v): # Controlla che ci sia qualcosa nelle liste di top_c
        messagebox.showinfo("Verifica Futura", "Nessun 'Top Combinato' dall'analisi precedente da verificare."); 
        risultato_text.config(state=tk.DISABLED); return
    try: n_colpi_fut = int(estrazioni_entry_verifica_futura.get()); assert 1 <= n_colpi_fut <= 50
    except: 
        messagebox.showerror("Input Invalido", "N. Colpi Verifica Futura (1-50) non valido."); 
        risultato_text.config(state=tk.DISABLED); return
    try: 
        res_str = verifica_esiti_futuri(top_c, nomi_rv, data_fine, n_colpi_fut)
        risultato_text.insert(tk.END, res_str)
    except Exception as e: 
        risultato_text.insert(tk.END, f"\nErrore durante la verifica esiti futuri: {e}"); traceback.print_exc()
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)

def esegui_verifica_mista():
    global info_ricerca_globale, risultato_text, root, text_combinazioni_miste, estrazioni_entry_verifica_mista
    risultato_text.config(state=tk.NORMAL)
    risultato_text.insert(tk.END, "\n\nVerifica mista (combinazioni utente su trigger spia)...")
    risultato_text.see(tk.END); root.update_idletasks()
    input_text = text_combinazioni_miste.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Input Invalido", "Nessuna combinazione inserita."); risultato_text.config(state=tk.DISABLED); return
    combinazioni_sets = {'estratto': set(),'ambo': set(),'terno': set(),'quaterna': set(),'cinquina': set()}
    righe_input_originali = []
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
    date_triggers = info_ricerca_globale.get('date_trigger_ordinate')
    nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date')
    end_ts = info_ricerca_globale.get('end_date')
    numeri_spia_usati = info_ricerca_globale.get('numeri_spia', [])
    if not all([date_triggers, nomi_rv, start_ts, end_ts]):
        messagebox.showerror("Errore Verifica Mista", "Dati analisi 'Successivi' (Date Trigger, Ruote Verifica, Periodo Analisi) mancanti."); 
        risultato_text.config(state=tk.DISABLED); return
    try: n_colpi_misti = int(estrazioni_entry_verifica_mista.get()); assert 1 <= n_colpi_misti <= 18
    except: messagebox.showerror("Input Invalido", "N. Colpi Verifica Mista (1-18) non valido."); risultato_text.config(state=tk.DISABLED); return
    try:
        titolo_output = f"VERIFICA MISTA (COMBINAZIONI UTENTE) - Dopo Spia: {', '.join(numeri_spia_usati)}"
        res_str = verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_rv, n_colpi_misti, start_ts, end_ts, titolo_sezione=titolo_output)
        summary_input = "\nInput utente originale:\n" + "\n".join([f"  - {r}" for r in righe_input_originali])
        lines = res_str.splitlines()
        insert_idx_summary = 1 
        final_output_lines = lines[:insert_idx_summary] + [summary_input] + lines[insert_idx_summary:]
        risultato_text.insert(tk.END, "\n".join(final_output_lines))
    except Exception as e: risultato_text.insert(tk.END, f"\nErrore durante la verifica mista: {e}"); traceback.print_exc()
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)

# =============================================================================
# Funzione Wrapper per Visualizza Grafici
# (OMESSA PER BREVITÀ - INVARIATA)
# =============================================================================
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale 
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and bool(risultati_globali) and any(r[2] for r in risultati_globali if len(r)>2):
        valid_res = [r for r in risultati_globali if r[2] is not None]
        if valid_res: visualizza_grafici(valid_res, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni',5))
        else: messagebox.showinfo("Grafici", "Nessun risultato valido per grafici.")
    else: messagebox.showinfo("Grafici", "Esegui 'Cerca Successivi' con risultati validi prima.")

# =============================================================================
# GUI e Mainloop
# (OMESSO PER BREVITÀ - È INVARIATO RISPETTO ALLA VERSIONE 3.8.8)
# =============================================================================
root = tk.Tk()
root.title("Analisi Lotto v3.8.12 - Codice Pulito") 
root.geometry("1350x850") 
root.minsize(1200, 750) 
root.configure(bg="#f0f0f0")

style = ttk.Style(); style.theme_use('clam')
style.configure("TFrame", background="#f0f0f0"); style.configure("TLabel", background="#f0f0f0", font=("Segoe UI",10))
style.configure("TButton", font=("Segoe UI",10), padding=5); style.configure("Title.TLabel", font=("Segoe UI",11,"bold"))
style.configure("Header.TLabel", font=("Segoe UI",12,"bold")); style.configure("Small.TLabel", background="#f0f0f0", font=("Segoe UI",8))
style.configure("TEntry", padding=3); style.configure("TListbox", font=("Consolas",10)); style.configure("TLabelframe.Label", font=("Segoe UI",10,"bold"), background="#f0f0f0")
style.configure("TNotebook.Tab", padding=[10,5], font=("Segoe UI",10))

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
ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi:", style="Title.TLabel").pack(anchor="w")
ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame); ruote_analisi_list_frame.pack(fill=tk.BOTH,expand=True)
scrollbar_ra = ttk.Scrollbar(ruote_analisi_list_frame); scrollbar_ra.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_ra.set)
listbox_ruote_analisi.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_ra.config(command=listbox_ruote_analisi.yview)

ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ); ruote_verifica_outer_frame.grid(row=0,column=1,sticky="nsew",padx=5)
ttk.Label(ruote_verifica_outer_frame, text="3. Ruote Verifica:", style="Title.TLabel").pack(anchor="w")
ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame); ruote_verifica_list_frame.pack(fill=tk.BOTH,expand=True)
scrollbar_rv = ttk.Scrollbar(ruote_verifica_list_frame); scrollbar_rv.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_rv.set)
listbox_ruote_verifica.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_rv.config(command=listbox_ruote_verifica.yview)

center_controls_frame_succ = ttk.Frame(controls_frame_succ); center_controls_frame_succ.grid(row=0,column=2,sticky="ns",padx=5)
spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Numeri Spia (1-90) ",padding=5); spia_frame_succ.pack(fill=tk.X,pady=(0,5))
spia_entry_container_succ = ttk.Frame(spia_frame_succ); spia_entry_container_succ.pack(fill=tk.X,pady=5)
entry_numeri_spia = [ttk.Entry(spia_entry_container_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)) for _ in range(5)]
for entry in entry_numeri_spia: entry.pack(side=tk.LEFT,padx=3,ipady=2)
estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 4. Estrazioni Successive ",padding=5); estrazioni_frame_succ.pack(fill=tk.X,pady=5)
ttk.Label(estrazioni_frame_succ, text="Quante (1-18):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_succ.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_succ.insert(0,"5")
verifica_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Verifica Esiti (Classica) ",padding=5); verifica_frame_succ.pack(fill=tk.X,pady=5)
ttk.Label(verifica_frame_succ, text="Estrazioni Verifica (1-18):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica = ttk.Entry(verifica_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica.insert(0,"9")

buttons_frame_succ = ttk.Frame(controls_frame_succ); buttons_frame_succ.grid(row=0,column=3,sticky="ns",padx=(10,0))
button_cerca_succ = ttk.Button(buttons_frame_succ, text="Cerca Successivi",command=lambda:cerca_numeri(modalita="successivi")); button_cerca_succ.pack(pady=5,fill=tk.X,ipady=3)
button_verifica_esiti = ttk.Button(buttons_frame_succ, text="Verifica Esiti\n(Classica)",command=esegui_verifica_esiti); button_verifica_esiti.pack(pady=5,fill=tk.X,ipady=0); button_verifica_esiti.config(state=tk.DISABLED)

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

ttk.Label(main_frame, text="Risultati Analisi:", style="Header.TLabel").pack(anchor="w",pady=(15,0))
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
    risultato_text.insert(tk.END, "Benvenuto!\n\n1. Usa 'Sfoglia...' per cartella estrazioni.\n2. Seleziona modalità e parametri.\n3. Imposta periodo analisi.\n4. Clicca 'Cerca...'.\n5. Dopo 'Cerca Successivi', usa:\n   - Verifica Esiti (Classica): per i top combinati.\n   - Grafici.\n   - Verifica Futura (Post-Analisi): per i top combinati, solo dopo data fine analisi.\n   - Verifica Mista (su Trigger Spia): per combinazioni utente, basata sui trigger spia.\n")
    risultato_text.config(state=tk.DISABLED)
    root.mainloop()
    print("\nScript terminato.")

if __name__ == "__main__":
    main()