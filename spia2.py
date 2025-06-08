# -*- coding: utf-8 -*-
# Versione 3.9.7 - ANALISI POSIZIONALE (SPIA + ESITI)
# MODIFICATO PER COPIA/INCOLLA E SALVATAGGIO POPUP

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

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}
MAX_COLPI_GIOCO = 90 # Nuovo limite per i colpi di gioco

# Variabili GUI globali
button_visualizza = None
button_verifica_futura = None
estrazioni_entry_verifica_futura = None
button_verifica_mista = None
text_combinazioni_miste = None
estrazioni_entry_verifica_mista = None

# MODIFICATO/NUOVO per analisi posizionale e GUI
tipo_spia_var_global = None
entry_numeri_spia = []
combo_posizione_spia = None

# Variabili GUI per Numeri Simpatici
entry_numero_target_simpatici = None
listbox_ruote_simpatici = None
entry_top_n_simpatici = None


# =============================================================================
# FUNZIONI GRAFICHE (Come da tua versione 3.9.3 - INVARIATE)
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
# FUNZIONI LOGICHE (INVARIATE NELLA LORO LOGICA PRINCIPALE)
# =============================================================================
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
            try:
                data_dt_val = datetime.datetime.strptime(data_str, fmt_ok)
                if len(nums_orig) != 5: continue
                [int(n) for n in nums_orig]
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
        df.dropna(subset=[f'Numero{i+1}' for i in range(5)], how='any', inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)
        return df if not df.empty else None
    except Exception as e:
        print(f"Errore lettura file {os.path.basename(file_path)}: {e}"); traceback.print_exc(); return None

def analizza_ruota_verifica(df_verifica, date_trigger_sorted, n_estrazioni, nome_ruota_verifica):
    if df_verifica is None or df_verifica.empty: return None, "Df verifica vuoto."
    df_verifica = df_verifica.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    n_trigger = len(date_trigger_sorted)
    if n_trigger == 0: # Aggiunto controllo per evitare divisione per zero se non ci sono trigger
        return {'totale_trigger': 0}, f"Nessun evento trigger per l'analisi su {nome_ruota_verifica}."

    date_series_verifica = df_verifica['Data']

    freq_estratti, freq_ambi, freq_terne = Counter(), Counter(), Counter()
    pres_estratti, pres_ambi, pres_terne = Counter(), Counter(), Counter()
    
    # Questo conteggio è per la sezione "Migliori Ambi per Copertura"
    # e non influenza direttamente il "Top per Presenza" o "Top per Frequenza" degli ambi
    # se non per il fatto che popola pres_ambi e freq_ambi.
    ambo_copertura_trigger = Counter() # Questo conta quante volte un ambo copre un trigger (max 1 per trigger)

    freq_pos_estratti = {}
    pres_pos_estratti = {}

    for data_t in date_trigger_sorted:
        try:
            start_index = date_series_verifica.searchsorted(data_t, side='right')
        except Exception:
            continue # Salta questo trigger se c'è un errore
        if start_index >= len(date_series_verifica):
            continue # Non ci sono estrazioni successive

        df_successive = df_verifica.iloc[start_index : start_index + n_estrazioni]
        
        # Set per tracciare gli item unici usciti *in questa finestra temporale* dopo *questo specifico trigger*
        # Questi sono usati per incrementare i contatori di *presenza* (pres_estratti, pres_ambi, pres_terne)
        estratti_unici_finestra, ambi_unici_finestra, terne_unici_finestra = set(), set(), set()
        estratti_pos_unici_finestra = {} # Per posizionale

        if not df_successive.empty:
            for _, row in df_successive.iterrows(): # Itera su ogni estrazione nella finestra successiva
                numeri_estratti_riga_con_pos = []
                for pos_idx, col_num_nome in enumerate(colonne_numeri):
                    num_val = row[col_num_nome]
                    if pd.notna(num_val):
                        numeri_estratti_riga_con_pos.append((str(num_val).zfill(2), pos_idx + 1))

                if not numeri_estratti_riga_con_pos: continue

                # Aggiorna Frequenze (conteggio totale di uscite)
                for num_str, pos in numeri_estratti_riga_con_pos:
                    freq_estratti[num_str] += 1
                    estratti_unici_finestra.add(num_str) # Per aggiornare pres_estratti dopo il loop sulla finestra

                    if num_str not in freq_pos_estratti:
                        freq_pos_estratti[num_str] = Counter()
                    freq_pos_estratti[num_str][pos] += 1

                    if num_str not in estratti_pos_unici_finestra:
                        estratti_pos_unici_finestra[num_str] = set()
                    estratti_pos_unici_finestra[num_str].add(pos)

                numeri_solo_per_combinazioni = sorted([item[0] for item in numeri_estratti_riga_con_pos])
                
                if len(numeri_solo_per_combinazioni) >= 2:
                    for ambo_tuple in itertools.combinations(numeri_solo_per_combinazioni, 2):
                        # ambo_tuple è già ordinato perché numeri_solo_per_combinazioni è ordinato
                        freq_ambi[ambo_tuple] += 1
                        ambi_unici_finestra.add(ambo_tuple) # Per aggiornare pres_ambi

                if len(numeri_solo_per_combinazioni) >= 3:
                    for terno_tuple in itertools.combinations(numeri_solo_per_combinazioni, 3):
                        # terno_tuple è già ordinato
                        freq_terne[terno_tuple] += 1
                        terne_unici_finestra.add(terno_tuple) # Per aggiornare pres_terne
        
        # Aggiorna Presenze (conteggio di quanti trigger sono stati "coperti")
        for num in estratti_unici_finestra: pres_estratti[num] += 1
        for ambo_u in ambi_unici_finestra:
            pres_ambi[ambo_u] += 1
            ambo_copertura_trigger[ambo_u] +=1 # Questo è il conteggio specifico per la sezione "Migliori Ambi per Copertura"
        for terno in terne_unici_finestra: pres_terne[terno] += 1
        
        for num_str, pos_set in estratti_pos_unici_finestra.items():
            if num_str not in pres_pos_estratti:
                pres_pos_estratti[num_str] = Counter()
            for pos_val in pos_set:
                pres_pos_estratti[num_str][pos_val] +=1

    results = {'totale_trigger': n_trigger}
    
    for tipo, freq_dict_raw, pres_dict_raw in [('estratto', freq_estratti, pres_estratti),
                                               ('ambo', freq_ambi, pres_ambi),
                                               ('terno', freq_terne, pres_terne)]:
        
        # Se non ci sono dati di frequenza (quindi nemmeno di presenza), inizializza e continua
        if not freq_dict_raw: # Se freq_dict_raw è vuoto, anche pres_dict_raw dovrebbe esserlo per come è costruito
            results[tipo] = {
                'presenza': {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'frequenze': pd.Series(dtype=int), 'perc_frequenza': pd.Series(dtype=float)},
                'frequenza': {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'presenze': pd.Series(dtype=int), 'perc_presenza': pd.Series(dtype=float)},
                'all_percentuali_presenza': pd.Series(dtype=float), 'all_percentuali_frequenza': pd.Series(dtype=float),
                'full_presenze': pd.Series(dtype=int), 'full_frequenze': pd.Series(dtype=int)
            }
            if tipo == 'estratto':
                 results[tipo].update({'posizionale_frequenza': {}, 'posizionale_presenza': {}})
            if tipo == 'ambo':
                results[tipo]['migliori_per_copertura_trigger'] = {'items': [], 'totale_trigger_spia': n_trigger}
            continue

        # Crea Series Pandas dai dizionari grezzi
        # Gli indici qui sono tuple per ambi/terni, stringhe per estratti
        freq_s_raw_idx = pd.Series(freq_dict_raw, dtype=int).sort_index()
        pres_s_raw_idx = pd.Series(pres_dict_raw, dtype=int)
        
        # Allinea pres_s_raw_idx con freq_s_raw_idx. Tutti gli item che hanno una frequenza > 0
        # dovrebbero avere una voce in pres_s_raw_idx (anche se 0, se non hanno coperto trigger)
        # e viceversa, se un item ha coperto un trigger, deve avere una frequenza > 0.
        # L'unione degli indici assicura che consideriamo tutti gli item che sono apparsi o hanno coperto.
        all_items_raw_idx = freq_s_raw_idx.index.union(pres_s_raw_idx.index)
        freq_s = freq_s_raw_idx.reindex(all_items_raw_idx, fill_value=0).sort_index()
        pres_s = pres_s_raw_idx.reindex(all_items_raw_idx, fill_value=0).sort_index()

        # Calcola percentuali
        tot_freq_val = freq_s.sum()
        perc_freq_s = (freq_s / tot_freq_val * 100).round(2) if tot_freq_val > 0 else pd.Series(0.0, index=freq_s.index, dtype=float)
        
        # n_trigger è la base per la percentuale di presenza
        perc_pres_s = (pres_s / n_trigger * 100).round(2) if n_trigger > 0 else pd.Series(0.0, index=pres_s.index, dtype=float)

        # --- Sezione "Top per Presenza" ---
        # Ordina per presenza (valore assoluto), poi per frequenza, poi per chiave per stabilità
        # if tipo == 'ambo': print(f"\nAmbi pres_s prima di top_pres:\n{pres_s[pres_s > 0].sort_values(ascending=False)}")
        
        # Prendiamo TUTTI gli item ordinati per presenza, poi frequenza, poi indice
        # Questo assicura che se ci sono molti item con presenza 0, non finiscano nei top se ce ne sono altri > 0
        sorted_by_pres = pres_s.sort_values(ascending=False)
        # Per stabilità e per rompere i pareggi in modo consistente
        # Creiamo un DataFrame temporaneo per ordinare su più colonne
        df_temp_pres = pd.DataFrame({'pres': pres_s, 'freq': freq_s})
        df_temp_pres = df_temp_pres.sort_values(by=['pres', 'freq'], ascending=[False, False])
        top_pres_items_series = df_temp_pres['pres'].head(10) # Questa è la Series dei valori di presenza dei top 10

        top_pres_items_data = []
        for idx_item, pres_val in top_pres_items_series.items():
            item_str = format_ambo_terno(idx_item) if tipo in ['ambo', 'terno'] else idx_item
            top_pres_items_data.append({
                'item': item_str,
                'valore': pres_val, # Presenza
                'percentuale': perc_pres_s.get(idx_item, 0.0),
                'altra_stat': freq_s.get(idx_item, 0), # Frequenza totale per questo item
                'altra_perc': perc_freq_s.get(idx_item, 0.0) # Percentuale di frequenza per questo item
            })
        
        # --- Sezione "Top per Frequenza" ---
        df_temp_freq = pd.DataFrame({'freq': freq_s, 'pres': pres_s})
        df_temp_freq = df_temp_freq.sort_values(by=['freq', 'pres'], ascending=[False, False])
        top_freq_items_series = df_temp_freq['freq'].head(10)

        top_freq_items_data = []
        for idx_item, freq_val in top_freq_items_series.items():
            item_str = format_ambo_terno(idx_item) if tipo in ['ambo', 'terno'] else idx_item
            top_freq_items_data.append({
                'item': item_str,
                'valore': freq_val, # Frequenza
                'percentuale': perc_freq_s.get(idx_item, 0.0),
                'altra_stat': pres_s.get(idx_item, 0), # Presenza per questo item
                'altra_perc': perc_pres_s.get(idx_item, 0.0) # Percentuale di presenza per questo item
            })

        results[tipo] = {
            'presenza': {
                'top_data': top_pres_items_data, # Lista di dizionari
                # Per compatibilità con grafici e output precedenti, ricreiamo le Series formattate dai top_data
                'top': pd.Series({d['item']: d['valore'] for d in top_pres_items_data}, dtype=int),
                'percentuali': pd.Series({d['item']: d['percentuale'] for d in top_pres_items_data}, dtype=float),
                'frequenze': pd.Series({d['item']: d['altra_stat'] for d in top_pres_items_data}, dtype=int),
                'perc_frequenza': pd.Series({d['item']: d['altra_perc'] for d in top_pres_items_data}, dtype=float)
            },
            'frequenza': {
                'top_data': top_freq_items_data, # Lista di dizionari
                'top': pd.Series({d['item']: d['valore'] for d in top_freq_items_data}, dtype=int),
                'percentuali': pd.Series({d['item']: d['percentuale'] for d in top_freq_items_data}, dtype=float),
                'presenze': pd.Series({d['item']: d['altra_stat'] for d in top_freq_items_data}, dtype=int),
                'perc_presenza': pd.Series({d['item']: d['altra_perc'] for d in top_freq_items_data}, dtype=float)
            },
            'all_percentuali_presenza': perc_pres_s.rename(index=lambda x: format_ambo_terno(x) if tipo in ['ambo','terno'] else x),
            'all_percentuali_frequenza': perc_freq_s.rename(index=lambda x: format_ambo_terno(x) if tipo in ['ambo','terno'] else x),
            'full_presenze': pres_s.rename(index=lambda x: format_ambo_terno(x) if tipo in ['ambo','terno'] else x),
            'full_frequenze': freq_s.rename(index=lambda x: format_ambo_terno(x) if tipo in ['ambo','terno'] else x)
        }
        # La formattazione degli indici per le Series 'top', 'percentuali', ecc. ora avviene usando
        # le chiavi 'item' già formattate da top_pres_items_data e top_freq_items_data.

    # Gestione 'migliori_per_copertura_trigger' per AMBO (come prima, ma basato su ambo_copertura_trigger)
    if 'ambo' in results and isinstance(results['ambo'], dict):
        if n_trigger > 0 and ambo_copertura_trigger: # Usa ambo_copertura_trigger qui
            migliori_ambi_copertura_trigger_raw = sorted(
                ambo_copertura_trigger.items(), # Questo contatore è specifico per questa sezione
                key=lambda item: (item[1], item[0]), # Ordina per conteggio, poi per ambo
                reverse=True
            )
            top_ambi_cop_list = []
            if migliori_ambi_copertura_trigger_raw:
                top_ambi_cop_list = [
                   (format_ambo_terno(ambo_tuple), count)
                   for ambo_tuple, count in migliori_ambi_copertura_trigger_raw
                ][:10] # Prendi i primi 10 per coerenza, o 3 se preferisci come prima

            results['ambo']['migliori_per_copertura_trigger'] = {
                'items': top_ambi_cop_list,
                'totale_trigger_spia': n_trigger
            }
        # else (se n_trigger <=0 o ambo_copertura_trigger è vuoto, già inizializzato sopra)

    # Gestione posizionale per ESTRATTO (come prima)
    if 'estratto' in results and isinstance(results['estratto'], dict): # Verifica aggiunta
        if freq_pos_estratti: # Controlla se non è vuoto
            results['estratto']['posizionale_frequenza'] = {
                num: dict(sorted(pos_counts.items())) for num, pos_counts in freq_pos_estratti.items()
            }
        if pres_pos_estratti: # Controlla se non è vuoto
            results['estratto']['posizionale_presenza'] = {
                num: dict(sorted(pos_counts.items())) for num, pos_counts in pres_pos_estratti.items()
            }
        # Se erano già stati inizializzati a {} e rimangono vuoti, va bene.

    return (results, None) if any(results.get(t) and results[t].get('full_frequenze') is not None and not results[t]['full_frequenze'].empty for t in ['estratto', 'ambo', 'terno']) else (None, f"Nessun risultato su {nome_ruota_verifica}.")

def analizza_antecedenti(df_ruota, numeri_obiettivo, n_precedenti, nome_ruota, top_n_candidati_copertura=15, max_k_copertura=5):
    if df_ruota is None or df_ruota.empty: return None, "DataFrame vuoto."
    if not numeri_obiettivo or n_precedenti <= 0: return None, "Input invalidi."

    df_ruota = df_ruota.sort_values(by='Data').reset_index(drop=True)
    cols_num = [f'Numero{i+1}' for i in range(5)]
    numeri_obiettivo_zfill = [str(n).zfill(2) for n in numeri_obiettivo]

    # --- RIGA CORRETTA/RIPRISTINATA ---
    indices_obiettivo = df_ruota.index[df_ruota[cols_num].isin(numeri_obiettivo_zfill).any(axis=1)].tolist()
    # --- FINE CORREZIONE ---
    
    n_occ_obiettivo_totali = len(indices_obiettivo)
    
    # Definizione di base_res spostata qui per essere accessibile anche in caso di errore precoce
    base_res = {
        'totale_occorrenze_obiettivo': n_occ_obiettivo_totali, 
        'base_presenza_antecedenti': 0, # Verrà aggiornato dopo
        'numeri_obiettivo': numeri_obiettivo_zfill, 
        'n_precedenti': n_precedenti, 
        'nome_ruota': nome_ruota
    }
    
    if n_occ_obiettivo_totali == 0:
        # Restituisci una struttura completa anche in caso di errore per coerenza
        empty_stats_val = empty_stats() # Assicurati che empty_stats sia definita o passala
        empty_freq_stats_val = empty_freq_stats() # Assicurati che empty_freq_stats sia definita o passala
        return {
            **base_res,
            'presenza': empty_stats_val,
            'frequenza': empty_freq_stats_val,
            'top_copertura_combinata_antecedenti': []
        }, f"Obiettivi {', '.join(numeri_obiettivo_zfill)} non trovati su {nome_ruota} nel periodo."

    freq_ant, pres_ant = Counter(), Counter()
    lista_set_antecedenti_per_caso = [] 

    for idx_obj in indices_obiettivo: # Ora 'indices_obiettivo' è definito
        if idx_obj < n_precedenti: 
            continue
        df_prec = df_ruota.iloc[idx_obj - n_precedenti : idx_obj]
        if not df_prec.empty:
            numeri_finestra_unici_per_questo_caso = set()
            numeri_per_freq_e_pres_individuale_in_finestra = []
            for _, row_prec in df_prec.iterrows():
                estratti_prec_riga = [row_prec[col] for col in cols_num if pd.notna(row_prec[col])]
                estratti_prec_riga_zfill = [str(n).zfill(2) for n in estratti_prec_riga]
                numeri_per_freq_e_pres_individuale_in_finestra.extend(estratti_prec_riga_zfill)
                numeri_finestra_unici_per_questo_caso.update(estratti_prec_riga_zfill)
            if numeri_finestra_unici_per_questo_caso:
                lista_set_antecedenti_per_caso.append(numeri_finestra_unici_per_questo_caso)
                pres_ant.update(list(numeri_finestra_unici_per_questo_caso))
            freq_ant.update(numeri_per_freq_e_pres_individuale_in_finestra)

    actual_base_pres_per_statistiche_individuali = len(lista_set_antecedenti_per_caso)
    base_res['base_presenza_antecedenti'] = actual_base_pres_per_statistiche_individuali # Aggiorna base_res

    empty_stats_val = {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'frequenze':pd.Series(dtype=int),'perc_frequenza':pd.Series(dtype=float)}
    empty_freq_stats_val = {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'presenze':pd.Series(dtype=int),'perc_presenza':pd.Series(dtype=float)}


    if actual_base_pres_per_statistiche_individuali == 0 or not freq_ant:
        return {
            **base_res, 
            'presenza': empty_stats_val, 
            'frequenza': empty_freq_stats_val,
            'top_copertura_combinata_antecedenti': []
        }, "Nessuna finestra/numero antecedente valido per le statistiche."
    
    ant_freq_s = pd.Series(freq_ant, dtype=int).sort_index()
    ant_pres_s = pd.Series({k: v for k,v in pres_ant.items() if k in ant_freq_s.index}, dtype=int)
    ant_pres_s = ant_pres_s.reindex(ant_freq_s.index, fill_value=0).sort_index()
    tot_ant_freq = ant_freq_s.sum()
    perc_ant_freq = (ant_freq_s / tot_ant_freq * 100).round(2) if tot_ant_freq > 0 else pd.Series(0.0, index=ant_freq_s.index)
    perc_ant_pres = (ant_pres_s / actual_base_pres_per_statistiche_individuali * 100).round(2) if actual_base_pres_per_statistiche_individuali > 0 else pd.Series(0.0, index=ant_pres_s.index)
    top_ant_pres = ant_pres_s.sort_values(ascending=False).head(10)
    top_ant_freq = ant_freq_s.sort_values(ascending=False).head(10)

    risultati_copertura_combinata_antecedenti = []
    if actual_base_pres_per_statistiche_individuali > 0 and lista_set_antecedenti_per_caso:
        candidati_per_combinazioni = [item for item, count in pres_ant.most_common(top_n_candidati_copertura)]
        if candidati_per_combinazioni:
            copertura_100_raggiunta = False
            for k in range(1, max_k_copertura + 1):
                if len(candidati_per_combinazioni) < k:
                    break 
                migliore_combinazione_k_numeri = None
                max_casi_coperti_k = -1
                for combo_numeri_tuple in itertools.combinations(candidati_per_combinazioni, k):
                    combo_numeri_set = set(combo_numeri_tuple)
                    casi_coperti_da_questa_combo = 0
                    for set_antecedenti_del_caso_corrente in lista_set_antecedenti_per_caso:
                        if not combo_numeri_set.isdisjoint(set_antecedenti_del_caso_corrente):
                            casi_coperti_da_questa_combo += 1
                    if casi_coperti_da_questa_combo > max_casi_coperti_k:
                        max_casi_coperti_k = casi_coperti_da_questa_combo
                        migliore_combinazione_k_numeri = sorted(list(combo_numeri_tuple))
                    elif casi_coperti_da_questa_combo == max_casi_coperti_k and migliore_combinazione_k_numeri:
                        if sorted(list(combo_numeri_tuple)) < migliore_combinazione_k_numeri:
                             migliore_combinazione_k_numeri = sorted(list(combo_numeri_tuple))
                if migliore_combinazione_k_numeri is not None and max_casi_coperti_k >= 0:
                    perc_copertura_k = (max_casi_coperti_k / actual_base_pres_per_statistiche_individuali * 100) if actual_base_pres_per_statistiche_individuali > 0 else 0.0
                    risultati_copertura_combinata_antecedenti.append({
                        "k": k,
                        "numeri": migliore_combinazione_k_numeri,
                        "casi_coperti": max_casi_coperti_k,
                        "percentuale_copertura": round(perc_copertura_k, 1)
                    })
                    if round(perc_copertura_k, 1) >= 100.0:
                        copertura_100_raggiunta = True
                        break 
                if copertura_100_raggiunta:
                    break
    return {
        **base_res, # Usa la base_res aggiornata
        'presenza': {'top':top_ant_pres, 
                     'percentuali':perc_ant_pres.reindex(top_ant_pres.index).fillna(0.0),
                     'frequenze':ant_freq_s.reindex(top_ant_pres.index).fillna(0).astype(int),
                     'perc_frequenza':perc_ant_freq.reindex(top_ant_pres.index).fillna(0.0)},
        'frequenza':{'top':top_ant_freq, 
                     'percentuali':perc_ant_freq.reindex(top_ant_freq.index).fillna(0.0),
                     'presenze':ant_pres_s.reindex(top_ant_freq.index).fillna(0).astype(int),
                     'perc_presenza':perc_ant_pres.reindex(top_ant_freq.index).fillna(0.0)},
        'top_copertura_combinata_antecedenti': risultati_copertura_combinata_antecedenti
    }, None

def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
    global risultati_globali, info_ricerca_globale, button_visualizza, button_verifica_futura, button_verifica_mista

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
        if has_end_date and has_ruote_verifica_info and button_verifica_futura: button_verifica_futura.config(state=tk.NORMAL)
        if has_date_trigger and has_ruote_verifica_info and button_verifica_mista: button_verifica_mista.config(state=tk.NORMAL)
    else:
        risultati_globali, info_ricerca_globale = [], {}

# MODIFICATO per non alterare permanentemente lo stato di risultato_text
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

def trova_migliore_combinazione_copertura(items_con_date_coperte, k_items, num_tot_eventi_spia,
                                        dizionario_ritardi_items_individuali=None,
                                        num_top_items_da_considerare=20):
    """
    Trova la migliore combinazione di k_items (es. terni) che massimizza la copertura degli eventi spia.
    Args:
        items_con_date_coperte (dict): Dizionario {item_tuple: set_date_coperte}.
        k_items (int): Numero di item da combinare (es. 3, 4, 5).
        num_tot_eventi_spia (int): Numero totale di eventi spia da coprire.
        dizionario_ritardi_items_individuali (dict, optional): Dizionario {item_tuple: ritardo_val}.
                                                                Se fornito, usato per arricchire l'output.
        num_top_items_da_considerare (int): Considera solo i primi N item (ordinati per copertura individuale)
                                            per formare le combinazioni. Limita la ricerca.
    Returns:
        dict: Informazioni sulla migliore combinazione trovata.
    """
    if not items_con_date_coperte or k_items <= 0 or num_tot_eventi_spia == 0:
        return {
            "items_combinati_dettagli": [], "items_combinati_str": [], "eventi_coperti": 0,
            "percentuale_copertura": 0.0, "totale_eventi_spia": num_tot_eventi_spia,
            "messaggio": "Input invalidi o nessun evento spia."
        }

    sorted_items_individuali = sorted(
        items_con_date_coperte.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    items_da_considerare_per_combinazioni = [item_info[0] for item_info in sorted_items_individuali[:num_top_items_da_considerare]]
    
    if len(items_da_considerare_per_combinazioni) < k_items:
        return {
            "items_combinati_dettagli": [], "items_combinati_str": [], "eventi_coperti": 0,
            "percentuale_copertura": 0.0, "totale_eventi_spia": num_tot_eventi_spia,
            "messaggio": f"Non ci sono abbastanza item unici (trovati {len(items_da_considerare_per_combinazioni)}) per formare combinazioni di {k_items}."
        }

    migliore_combinazione_attuale_tuple_ordinata = None # Conterrà la tupla di tuple, ordinata
    max_copertura_attuale = -1

    for combo_tuple_di_items_raw in itertools.combinations(items_da_considerare_per_combinazioni, k_items):
        # Ordina la combinazione stessa (es. i 3 terni) per una rappresentazione canonica
        combo_tuple_di_items = tuple(sorted(list(combo_tuple_di_items_raw), key=lambda t: tuple(map(str,t))))


        date_coperte_dalla_combinazione = set()
        for item_in_combo in combo_tuple_di_items: # item_in_combo è una tupla (es. un terno)
            date_coperte_dalla_combinazione.update(items_con_date_coperte.get(item_in_combo, set()))
        
        copertura_combinazione = len(date_coperte_dalla_combinazione)

        if copertura_combinazione > max_copertura_attuale:
            max_copertura_attuale = copertura_combinazione
            migliore_combinazione_attuale_tuple_ordinata = combo_tuple_di_items
        elif copertura_combinazione == max_copertura_attuale:
            # Se la copertura è uguale, preferisci la combinazione "più piccola" lessicograficamente
            if migliore_combinazione_attuale_tuple_ordinata is None or combo_tuple_di_items < migliore_combinazione_attuale_tuple_ordinata:
                 migliore_combinazione_attuale_tuple_ordinata = combo_tuple_di_items


    if migliore_combinazione_attuale_tuple_ordinata is None:
         return {
            "items_combinati_dettagli": [], "items_combinati_str": [], "eventi_coperti": 0,
            "percentuale_copertura": 0.0, "totale_eventi_spia": num_tot_eventi_spia,
            "messaggio": f"Nessuna combinazione di {k_items} item trovata."
        }

    items_combinati_str_list = sorted([format_ambo_terno(item) for item in migliore_combinazione_attuale_tuple_ordinata])
    items_combinati_dettagli_list = []
    
    # Usa migliore_combinazione_attuale_tuple_ordinata che contiene le tuple originali degli item
    for item_tuple_originale in migliore_combinazione_attuale_tuple_ordinata:
        item_str = format_ambo_terno(item_tuple_originale)
        ritardo_display = ""
        if dizionario_ritardi_items_individuali:
            ritardo_val = dizionario_ritardi_items_individuali.get(item_tuple_originale)
            if ritardo_val is not None and ritardo_val not in ["N/A", "N/D"] and isinstance(ritardo_val, (int, float)):
                ritardo_display = f" [Rit.Min.Att: {int(ritardo_val)}]"
            elif ritardo_val == "N/D": # Gestisce esplicitamente "N/D"
                ritardo_display = " [Rit.Min.Att: N/D]"
            # Se ritardo_val è None o "N/A", ritardo_display rimane vuoto
        items_combinati_dettagli_list.append(f"{item_str}{ritardo_display}")
    
    # Assicurati che items_combinati_dettagli_list sia ordinata come items_combinati_str_list se necessario per coerenza di output
    # Dato che migliore_combinazione_attuale_tuple_ordinata è già ordinata, e iteriamo su quella, dovrebbe andare bene.
    # Se si vuole un ordinamento stringa finale per items_combinati_dettagli_list:
    items_combinati_dettagli_list.sort()


    percentuale = (max_copertura_attuale / num_tot_eventi_spia * 100) if num_tot_eventi_spia > 0 else 0.0

    return {
        "items_combinati_dettagli": items_combinati_dettagli_list,
        "items_combinati_str": items_combinati_str_list,
        "eventi_coperti": max_copertura_attuale,
        "percentuale_copertura": percentuale,
        "totale_eventi_spia": num_tot_eventi_spia,
        "messaggio": None
    }

def calcola_ritardo_attuale(df_ruota_completa, item_da_cercare, tipo_item, data_fine_analisi):
    """
    Calcola il ritardo attuale di un estratto, ambo o terno su una specifica ruota
    fino a una data di fine analisi.
    """
    if df_ruota_completa is None or df_ruota_completa.empty:
        return "N/D (no data)"
    if not isinstance(data_fine_analisi, pd.Timestamp):
        data_fine_analisi = pd.Timestamp(data_fine_analisi)

    # Filtra le estrazioni fino alla data di fine analisi e ordina per data decrescente
    df_filtrato = df_ruota_completa[df_ruota_completa['Data'] <= data_fine_analisi].sort_values(by='Data', ascending=False)

    if df_filtrato.empty:
        # Se non ci sono estrazioni nel periodo fino a data_fine_analisi,
        # potremmo considerare il ritardo come il numero totale di estrazioni in df_ruota_completa
        # oppure "N/D". Per coerenza con il comportamento precedente, se df_filtrato è vuoto,
        # significa che l'item non è mai uscito (o non ci sono dati) *fino a quella data*.
        # Se df_ruota_completa non è vuoto, allora il ritardo è la sua lunghezza.
        # Ma se la richiesta è "fino a data_fine_analisi", e non ci sono estrazioni lì, è più corretto N/D.
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
                # Assicurati che item_da_cercare sia una tupla di stringhe zfillate
                item_ambo_set = set(str(n).zfill(2) for n in item_da_cercare)
                if item_ambo_set.issubset(numeri_riga_set):
                    trovato = True
        elif tipo_item == "terno": # <<< NUOVA PARTE >>>
            if isinstance(item_da_cercare, tuple) and len(item_da_cercare) == 3:
                # Assicurati che item_da_cercare sia una tupla di stringhe zfillate
                item_terno_set = set(str(n).zfill(2) for n in item_da_cercare)
                if item_terno_set.issubset(numeri_riga_set):
                    trovato = True
        # Puoi aggiungere qui 'quaterna', 'cinquina' in futuro se necessario

        if trovato:
            return ritardo -1 # -1 perché il ritardo è 0 se esce nell'ultima estrazione considerata

    # Se non trovato dopo aver iterato tutte le estrazioni in df_filtrato,
    # il ritardo è il numero di estrazioni in df_filtrato.
    return ritardo


# MODIFICATO per includere pulsante "Salva su File..." e correggere struttura
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
    text_widget.config(state=tk.DISABLED) # Permette la selezione e la copia standard

    button_frame_popup = ttk.Frame(popup_window)
    button_frame_popup.pack(fill=tk.X, pady=(0,10), padx=10, side=tk.BOTTOM)

def mostra_popup_risultati_spia(info_ricerca, risultati_analisi):
    global root

    popup = tk.Toplevel(root)
    popup.title("Riepilogo Analisi Numeri Spia")
    popup.geometry("850x800")
    popup.transient(root)

    text_area_popup = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
    text_area_popup.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10,0))

    popup_content_list = ["=== RIEPILOGO ANALISI NUMERI SPIA ===\n"]
    # ... (parte iniziale per tipo spia, ruote, periodo, ecc. - INVARIATA) ...
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
    date_eventi_spia = info_ricerca.get('date_trigger_ordinate', [])
    popup_content_list.append(f"Numero Totale di Eventi Spia: {len(date_eventi_spia)}")
    popup_content_list.append("-" * 60)


    if not risultati_analisi and not date_eventi_spia:
        popup_content_list.append("\nNessun evento spia trovato nel periodo e con i criteri specificati.")
    elif not risultati_analisi and date_eventi_spia :
        popup_content_list.append("\nNessun risultato dettagliato per le ruote di verifica (controllare selezione ruote o dati).")
    else:
        for nome_ruota_v, _, res_ruota in risultati_analisi:
            if not res_ruota or not isinstance(res_ruota, dict): continue
            popup_content_list.append(f"\n\n--- RISULTATI PER RUOTA DI VERIFICA: {nome_ruota_v.upper()} ---")
            num_eventi_ruota = res_ruota.get('totale_trigger', res_ruota.get('totale_eventi_spia', len(date_eventi_spia)))
            popup_content_list.append(f"(Basato su {num_eventi_ruota} Eventi Spia)")
            
            for tipo_esito in ['estratto', 'ambo', 'terno']:
                dati_esito = res_ruota.get(tipo_esito)
                if dati_esito:
                    popup_content_list.append(f"\n  -- {tipo_esito.capitalize()} Successivi --")
                    popup_content_list.append(f"    Top per Presenza (su {num_eventi_ruota} Eventi Spia):")
                    
                    top_pres_items_list = dati_esito.get('presenza', {}).get('top_data')
                    if top_pres_items_list:
                        for data_item in top_pres_items_list:
                            item_str_key_popup = data_item['item']
                            pres_val_popup = data_item['valore']
                            perc_popup = data_item['percentuale']
                            freq_popup = data_item['altra_stat']
                            riga_base_popup = f"      - {item_str_key_popup}: {pres_val_popup} ({perc_popup:.1f}%) [Freq.Tot: {freq_popup}]"; ritardo_str_popup = ""
                            if tipo_esito in ['estratto', 'ambo', 'terno'] and 'ritardi_attuali' in dati_esito:
                                ritardo_val_popup = dati_esito['ritardi_attuali'].get(item_str_key_popup)
                                if ritardo_val_popup is not None and ritardo_val_popup not in ["N/A (parse)", "N/A (err)", "N/D (no data full)", "N/D"]: ritardo_str_popup = f" [Rit.Att: {ritardo_val_popup}]"
                                elif ritardo_val_popup: ritardo_str_popup = f" [Rit.Att: {ritardo_val_popup}]"
                                else: ritardo_str_popup = f" [Rit.Att: N/D]"
                            popup_content_list.append(riga_base_popup + ritardo_str_popup)
                            if tipo_esito == 'estratto' and 'posizionale_presenza' in dati_esito and item_str_key_popup in dati_esito['posizionale_presenza']:
                                pos_data = dati_esito['posizionale_presenza'][item_str_key_popup]; pos_str_list = []
                                for pos_num in sorted(pos_data.keys()): pos_count = pos_data[pos_num]; pos_perc = (pos_count / pres_val_popup * 100) if pres_val_popup > 0 else 0; pos_str_list.append(f"P{pos_num}:{pos_count}({pos_perc:.0f}%)")
                                if pos_str_list: popup_content_list.append(f"        Posizioni (Pres.): {', '.join(pos_str_list)}")
                    elif dati_esito.get('presenza', {}).get('top') is not None and not dati_esito['presenza']['top'].empty: 
                        top_pres = dati_esito['presenza']['top']
                        for item_str_key_popup, pres_val_popup in top_pres.items():
                            perc_popup = dati_esito['presenza']['percentuali'].get(item_str_key_popup, 0.0); freq_popup = dati_esito['presenza']['frequenze'].get(item_str_key_popup, 0)
                            riga_base_popup = f"      - {item_str_key_popup}: {pres_val_popup} ({perc_popup:.1f}%) [Freq.Tot: {freq_popup}]"; ritardo_str_popup = ""
                            if tipo_esito in ['estratto', 'ambo', 'terno'] and 'ritardi_attuali' in dati_esito:
                                ritardo_val_popup = dati_esito['ritardi_attuali'].get(item_str_key_popup)
                                if ritardo_val_popup is not None and ritardo_val_popup not in ["N/A (parse)", "N/A (err)", "N/D (no data full)", "N/D"]: ritardo_str_popup = f" [Rit.Att: {ritardo_val_popup}]"
                                elif ritardo_val_popup: ritardo_str_popup = f" [Rit.Att: {ritardo_val_popup}]"
                                else: ritardo_str_popup = f" [Rit.Att: N/D]"
                            popup_content_list.append(riga_base_popup + ritardo_str_popup)
                            if tipo_esito == 'estratto' and 'posizionale_presenza' in dati_esito and item_str_key_popup in dati_esito['posizionale_presenza']:
                                pos_data = dati_esito['posizionale_presenza'][item_str_key_popup]; pos_str_list = []
                                for pos_num in sorted(pos_data.keys()): pos_count = pos_data[pos_num]; pos_perc = (pos_count / pres_val_popup * 100) if pres_val_popup > 0 else 0; pos_str_list.append(f"P{pos_num}:{pos_count}({pos_perc:.0f}%)")
                                if pos_str_list: popup_content_list.append(f"        Posizioni (Pres.): {', '.join(pos_str_list)}")
                    else: popup_content_list.append("      Nessuno.")
                    
                    popup_content_list.append(f"    Top per Frequenza (su {num_eventi_ruota} Eventi Spia):")
                    top_freq_items_list = dati_esito.get('frequenza', {}).get('top_data')
                    if top_freq_items_list:
                        for data_item_f in top_freq_items_list:
                            item_str_key_popup_f = data_item_f['item']
                            freq_val_popup_f = data_item_f['valore']
                            perc_popup_f = data_item_f['percentuale']
                            pres_popup_f = data_item_f['altra_stat']
                            riga_base_popup_f = f"      - {item_str_key_popup_f}: {freq_val_popup_f} ({perc_popup_f:.1f}%) [Pres. su Eventi Spia: {pres_popup_f}]"; popup_content_list.append(riga_base_popup_f)
                            if tipo_esito == 'estratto' and 'posizionale_frequenza' in dati_esito and item_str_key_popup_f in dati_esito['posizionale_frequenza']:
                                pos_data_f = dati_esito['posizionale_frequenza'][item_str_key_popup_f]; pos_str_list_f = []
                                for pos_num_f in sorted(pos_data_f.keys()): pos_count_f = pos_data_f[pos_num_f]; pos_perc_f = (pos_count_f / freq_val_popup_f * 100) if freq_val_popup_f > 0 else 0; pos_str_list_f.append(f"P{pos_num_f}:{pos_count_f}({pos_perc_f:.0f}%)")
                                if pos_str_list_f: popup_content_list.append(f"        Posizioni (Freq.): {', '.join(pos_str_list_f)}")
                    elif dati_esito.get('frequenza', {}).get('top') is not None and not dati_esito['frequenza']['top'].empty:
                        top_freq = dati_esito['frequenza']['top']
                        for item_str_key_popup_f, freq_val_popup_f in top_freq.items():
                            perc_popup_f = dati_esito['frequenza']['percentuali'].get(item_str_key_popup_f, 0.0); pres_popup_f = dati_esito['frequenza']['presenze'].get(item_str_key_popup_f, 0)
                            riga_base_popup_f = f"      - {item_str_key_popup_f}: {freq_val_popup_f} ({perc_popup_f:.1f}%) [Pres. su Eventi Spia: {pres_popup_f}]"; popup_content_list.append(riga_base_popup_f)
                            if tipo_esito == 'estratto' and 'posizionale_frequenza' in dati_esito and item_str_key_popup_f in dati_esito['posizionale_frequenza']:
                                pos_data_f = dati_esito['posizionale_frequenza'][item_str_key_popup_f]; pos_str_list_f = []
                                for pos_num_f in sorted(pos_data_f.keys()): pos_count_f = pos_data_f[pos_num_f]; pos_perc_f = (pos_count_f / freq_val_popup_f * 100) if freq_val_popup_f > 0 else 0; pos_str_list_f.append(f"P{pos_num_f}:{pos_count_f}({pos_perc_f:.0f}%)")
                                if pos_str_list_f: popup_content_list.append(f"        Posizioni (Freq.): {', '.join(pos_str_list_f)}")
                    else: popup_content_list.append("      Nessuno.")

                    # --- SEZIONE RIMOSSA ---
                    # if tipo_esito == 'ambo':
                    #     migliori_ambi_cop_info = dati_esito.get('migliori_per_copertura_trigger')
                    #     if migliori_ambi_cop_info and migliori_ambi_cop_info['items']:
                    #         popup_content_list.append(f"    Migliori Ambi per Copertura Eventi Spia (su {migliori_ambi_cop_info.get('totale_trigger_spia', num_eventi_ruota)} totali):")
                    #         for ambo_str_popup_c, count_cop_popup_c in migliori_ambi_cop_info['items']:
                    #             perc_cop_popup_c = (count_cop_popup_c / migliori_ambi_cop_info.get('totale_trigger_spia', num_eventi_ruota) * 100) if migliori_ambi_cop_info.get('totale_trigger_spia', num_eventi_ruota) > 0 else 0
                    #             popup_content_list.append(f"      - Ambo {ambo_str_popup_c}: Coperti {count_cop_popup_c} eventi spia ({perc_cop_popup_c:.1f}%)")
                    #     elif dati_esito: popup_content_list.append(f"    Migliori Ambi per Copertura Eventi Spia: Nessuno con copertura significativa.")
                    # --- FINE SEZIONE RIMOSSA ---
                else: popup_content_list.append(f"\n  -- {tipo_esito.capitalize()} Successivi: Nessun dato trovato.")

    # ... (resto della funzione per RISULTATI COMBINATI, ESTRATTI GLOBALI, AMBI GLOBALI, TERNI GLOBALI - INVARIATO) ...
    # (Assicurati che la terminologia "Eventi Spia" sia usata consistentemente anche in queste sezioni successive)
    statistiche_combinate_dett = info_ricerca.get('statistiche_combinate_dettagliate')
    if statistiche_combinate_dett and any(v for v in statistiche_combinate_dett.values() if v):
        popup_content_list.append("\n\n" + "=" * 25 + " RISULTATI COMBINATI (PER PUNTEGGIO) " + "=" * 25)
        for tipo_esito_comb in ['estratto', 'ambo', 'terno']:
            dati_tipo_comb_dett = statistiche_combinate_dett.get(tipo_esito_comb)
            if dati_tipo_comb_dett:
                popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati (per Punteggio) --")
                for i, stat_item in enumerate(dati_tipo_comb_dett):
                    item_str_c = stat_item["item"]; score_c = stat_item["punteggio"]; pres_avg_c = stat_item["presenza_media_perc"]; freq_tot_c = stat_item["frequenza_totale"]; ritardo_comb_str = ""
                    if tipo_esito_comb in ['estratto', 'ambo', 'terno'] and "ritardo_min_attuale" in stat_item: 
                        rit_val = stat_item["ritardo_min_attuale"]
                        if rit_val not in ["N/A", "N/D", None] and isinstance(rit_val, (int,float)): ritardo_comb_str = f" [Rit.Min.Att: {int(rit_val)}]"
                        elif rit_val == "N/D": ritardo_comb_str = f" [Rit.Min.Att: N/D]"
                    popup_content_list.append(f"    {i+1}. {item_str_c}: Punt={score_c:.2f} (PresAvg:{pres_avg_c:.1f}%, FreqTot:{freq_tot_c}){ritardo_comb_str}")
            else: popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati: Nessuno.")
    elif info_ricerca.get('top_combinati') and any(v for v in info_ricerca.get('top_combinati').values() if v) :
        popup_content_list.append("\n\n" + "=" * 25 + " RISULTATI COMBINATI (SOLO ITEM) " + "=" * 25)
        top_combinati_fallback = info_ricerca.get('top_combinati')
        for tipo_esito_comb_f in ['estratto', 'ambo', 'terno']:
            if top_combinati_fallback.get(tipo_esito_comb_f):
                popup_content_list.append(f"\n  -- Top {tipo_esito_comb_f.capitalize()} Combinati (solo item) --")
                for item_comb_f in top_combinati_fallback[tipo_esito_comb_f][:10]: popup_content_list.append(f"    - {item_comb_f}")
            else: popup_content_list.append(f"\n  -- Top {tipo_esito_comb_f.capitalize()} Combinati: Nessuno.")

    migliori_estratti_globali_info = info_ricerca.get('migliori_estratti_copertura_globale')
    if migliori_estratti_globali_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI ESTRATTI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append(f"(Uscita su QUALSIASI ruota di verifica dopo ogni Evento Spia)")
        for i, estratto_info_popup in enumerate(migliori_estratti_globali_info):
            rit_glob_str_popup_est = ""
            if "ritardo_min_attuale" in estratto_info_popup:
                rit_val_glob_popup_est = estratto_info_popup["ritardo_min_attuale"]
                if rit_val_glob_popup_est is not None and rit_val_glob_popup_est not in ["N/A", "N/D"] and isinstance(rit_val_glob_popup_est, (int,float)):
                    rit_glob_str_popup_est = f" [Rit.Min.Att: {int(rit_val_glob_popup_est)}]"
                elif rit_val_glob_popup_est == "N/D":
                    rit_glob_str_popup_est = " [Rit.Min.Att: N/D]"
            popup_content_list.append(f"  {i+1}. Estratto {estratto_info_popup['estratto']}: Coperti {estratto_info_popup['coperti']} su {estratto_info_popup['totali']} Eventi Spia ({estratto_info_popup['percentuale']:.1f}%){rit_glob_str_popup_est}")
    elif 'migliori_estratti_copertura_globale' in info_ricerca:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI ESTRATTI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append("  Nessun estratto individuale con copertura globale significativa trovato.")

    combinazione_ottimale_estratti_info = info_ricerca.get('combinazione_ottimale_estratti_100')
    migliore_parziale_estratti_info = info_ricerca.get('migliore_combinazione_parziale_estratti')
    if combinazione_ottimale_estratti_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORE COPERTURA COMBINATA ESTRATTI (100%) " + "=" * 10)
        num_estr_comb_ott = combinazione_ottimale_estratti_info.get('num_estratti_nella_combinazione', len(combinazione_ottimale_estratti_info.get('estratti_dettagli', [])))
        popup_content_list.append(f"  I seguenti {num_estr_comb_ott} estratto/i:")
        estratti_da_mostrare_ottimale = combinazione_ottimale_estratti_info.get("estratti_dettagli", [])
        for estratto_s_combinazione_dett in estratti_da_mostrare_ottimale:
            popup_content_list.append(f"    - {estratto_s_combinazione_dett}")
        popup_content_list.append(f"  Insieme hanno coperto il 100% ({combinazione_ottimale_estratti_info['coperti']}/{combinazione_ottimale_estratti_info['totali']} Eventi Spia).")
    elif migliore_parziale_estratti_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORE COPERTURA COMBINATA ESTRATTI (NON 100%) " + "=" * 10)
        num_estr_comb_parz = migliore_parziale_estratti_info.get('num_estratti_nella_combinazione', len(migliore_parziale_estratti_info.get('estratti_dettagli', [])))
        popup_content_list.append(f"  Nessuna combinazione di 1 fino a {num_estr_comb_parz} estratti (dai top) ha raggiunto il 100%.")
        estratti_da_visualizzare_popup_parziale = migliore_parziale_estratti_info.get("estratti_dettagli", [])
        popup_content_list.append(f"  Considerando la migliore combinazione di {num_estr_comb_parz} estratti:")
        for estratto_s_parziale_dett in estratti_da_visualizzare_popup_parziale:
            popup_content_list.append(f"    - {estratto_s_parziale_dett}")
        popup_content_list.append(f"  Copertura combinata: {migliore_parziale_estratti_info['coperti']}/{migliore_parziale_estratti_info['totali']} Eventi Spia ({migliore_parziale_estratti_info['percentuale']:.1f}%).")
    elif 'migliori_estratti_copertura_globale' in info_ricerca and info_ricerca['migliori_estratti_copertura_globale'] is not None \
         and not combinazione_ottimale_estratti_info and not migliore_parziale_estratti_info \
         and 'date_trigger_ordinate' in info_ricerca and len(info_ricerca['date_trigger_ordinate']) > 0 :
        popup_content_list.append("\n\n" + "=" * 10 + " COMBINAZIONE ESTRATTI PER COPERTURA " + "=" * 10)
        popup_content_list.append("  Non è stato possibile trovare una combinazione di estratti per la copertura totale,")
        popup_content_list.append("  o non sono stati trovati estratti con copertura globale sufficiente per la ricerca combinata.")
    
    migliori_ambi_globali_info = info_ricerca.get('migliori_ambi_copertura_globale')
    if migliori_ambi_globali_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI AMBI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append(f"(Uscita su QUALSIASI ruota di verifica dopo ogni Evento Spia)")
        for i, ambo_info_popup in enumerate(migliori_ambi_globali_info):
            rit_glob_str_popup = ""
            if "ritardo_min_attuale" in ambo_info_popup:
                rit_val_glob_popup = ambo_info_popup["ritardo_min_attuale"]
                if rit_val_glob_popup is not None and rit_val_glob_popup not in ["N/A", "N/D"] and isinstance(rit_val_glob_popup, (int,float)): rit_glob_str_popup = f" [Rit.Min.Att: {int(rit_val_glob_popup)}]"
                elif rit_val_glob_popup == "N/D": rit_glob_str_popup = f" [Rit.Min.Att: N/D]"
            popup_content_list.append(f"  {i+1}. Ambo {ambo_info_popup['ambo']}: Coperti {ambo_info_popup['coperti']} su {ambo_info_popup['totali']} Eventi Spia ({ambo_info_popup['percentuale']:.1f}%){rit_glob_str_popup}")
    elif 'migliori_ambi_copertura_globale' in info_ricerca:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI AMBI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append("  Nessun ambo individuale con copertura globale significativa trovato.")

    combinazione_ottimale_ambi_info = info_ricerca.get('combinazione_ottimale_copertura_100')
    migliore_parziale_ambi_info = info_ricerca.get('migliore_combinazione_parziale')
    if combinazione_ottimale_ambi_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORE COPERTURA COMBINATA AMBI (100%) " + "=" * 10)
        popup_content_list.append(f"  I seguenti {len(combinazione_ottimale_ambi_info.get('ambi_dettagli', combinazione_ottimale_ambi_info.get('ambi',[])))} ambo/i:")
        ambi_da_mostrare_ottimale = combinazione_ottimale_ambi_info.get("ambi_dettagli", combinazione_ottimale_ambi_info.get("ambi",[]))
        for ambo_s_combinazione_dett in ambi_da_mostrare_ottimale: popup_content_list.append(f"    - {ambo_s_combinazione_dett}")
        popup_content_list.append(f"  Insieme hanno coperto il 100% ({combinazione_ottimale_ambi_info['coperti']}/{combinazione_ottimale_ambi_info['totali']} Eventi Spia).")
    elif migliore_parziale_ambi_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORE COPERTURA COMBINATA AMBI (NON 100%) " + "=" * 10)
        popup_content_list.append(f"  Nessuna combinazione di 1, 2 o 3 ambi (dai top) ha raggiunto il 100%.")
        ambi_da_visualizzare_popup_parziale = migliore_parziale_ambi_info.get("ambi_dettagli", migliore_parziale_ambi_info.get("ambi",[]))
        popup_content_list.append(f"  Considerando i Top {len(migliore_parziale_ambi_info.get('ambi',[]))} ambi:")
        for ambo_s_parziale_dett in ambi_da_visualizzare_popup_parziale: popup_content_list.append(f"    - {ambo_s_parziale_dett}")
        popup_content_list.append(f"  Copertura combinata: {migliore_parziale_ambi_info['coperti']}/{migliore_parziale_ambi_info['totali']} Eventi Spia ({migliore_parziale_ambi_info['percentuale']:.1f}%).")
    elif 'migliori_ambi_copertura_globale' in info_ricerca and info_ricerca['migliori_ambi_copertura_globale'] is not None and not combinazione_ottimale_ambi_info and not migliore_parziale_ambi_info and 'date_trigger_ordinate' in info_ricerca and len(info_ricerca['date_trigger_ordinate']) > 0 :
        popup_content_list.append("\n\n" + "=" * 10 + " COMBINAZIONE AMBI PER COPERTURA " + "=" * 10)
        popup_content_list.append("  Non è stato possibile trovare una combinazione di 1-3 ambi (dai top) per la copertura totale,")
        popup_content_list.append("  o non sono stati trovati ambi con copertura globale sufficiente per la ricerca combinata.")
    
    migliori_terni_individuali_info_popup = info_ricerca.get('migliori_terni_copertura_globale')
    if migliori_terni_individuali_info_popup:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI TERNI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append(f"(Uscita su QUALSIASI ruota di verifica dopo ogni Evento Spia)")
        for i, terno_info_p in enumerate(migliori_terni_individuali_info_popup):
            rit_glob_str_p_terno = ""
            if "ritardo_min_attuale" in terno_info_p:
                rit_val_glob_p_terno = terno_info_p["ritardo_min_attuale"]
                if rit_val_glob_p_terno is not None and rit_val_glob_p_terno not in ["N/A", "N/D"] and isinstance(rit_val_glob_p_terno, (int,float)): rit_glob_str_p_terno = f" [Rit.Min.Att: {int(rit_val_glob_p_terno)}]"
                elif rit_val_glob_p_terno == "N/D": rit_glob_str_p_terno = " [Rit.Min.Att: N/D]"
            popup_content_list.append(f"  {i+1}. Terno {terno_info_p['terno']}: Coperti {terno_info_p['coperti']} su {terno_info_p['totali']} Eventi Spia ({terno_info_p['percentuale']:.1f}%){rit_glob_str_p_terno}")
    elif 'migliori_terni_copertura_globale' in info_ricerca:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI TERNI INDIVIDUALI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append("  Nessun terno individuale con copertura globale significativa trovato.")

    migliori_terni_combinati_output_list_popup = []
    titolo_terni_combinati_stampato_popup = False
    config_comb_terni_popup = info_ricerca.get('config_combinazioni_terni', {}); num_usati_str_popup = ""
    if config_comb_terni_popup:
        num_considerati = config_comb_terni_popup.get('num_top_terni_considerati')
        if num_considerati: num_usati_str_popup = f" (Ricerca tra Top {num_considerati} terni individuali)"
    
    migliore_k_per_popup_terni = None; max_copertura_per_popup_terni = -1
    risultato_miglior_k_terni_popup = None
    for k_val_check in [3, 4, 5]:
        info_key_check = f'migliore_combinazione_{k_val_check}_terni'
        combinazione_check = info_ricerca.get(info_key_check)
        if combinazione_check:
            copertura_check = combinazione_check.get("eventi_coperti", -1)
            if copertura_check > max_copertura_per_popup_terni:
                max_copertura_per_popup_terni = copertura_check; migliore_k_per_popup_terni = k_val_check
                risultato_miglior_k_terni_popup = combinazione_check
            elif copertura_check == max_copertura_per_popup_terni and migliore_k_per_popup_terni is None:
                migliore_k_per_popup_terni = k_val_check; risultato_miglior_k_terni_popup = combinazione_check
            elif copertura_check == max_copertura_per_popup_terni and k_val_check < migliore_k_per_popup_terni:
                migliore_k_per_popup_terni = k_val_check
                risultato_miglior_k_terni_popup = combinazione_check
    
    if risultato_miglior_k_terni_popup and migliore_k_per_popup_terni is not None:
        if not titolo_terni_combinati_stampato_popup:
            migliori_terni_combinati_output_list_popup.append("\n\n" + "=" * 10 + f" MIGLIORE COPERTURA COMBINATA TERNI " + "=" * 10)
            migliori_terni_combinati_output_list_popup.append(f"(Eventi Spia totali: {risultato_miglior_k_terni_popup.get('totale_eventi_spia', 'N/D')}){num_usati_str_popup}")
            titolo_terni_combinati_stampato_popup = True
        migliori_terni_combinati_output_list_popup.append(f"\n  -- Migliore Combinazione di {migliore_k_per_popup_terni} Terni --")
        messaggio_specifico = risultato_miglior_k_terni_popup.get("messaggio")
        if messaggio_specifico: migliori_terni_combinati_output_list_popup.append(f"    {messaggio_specifico}")
        items_combinati_popup = risultato_miglior_k_terni_popup.get("items_combinati_dettagli", [])
        eventi_coperti_da_combo = risultato_miglior_k_terni_popup.get("eventi_coperti", 0)
        if items_combinati_popup and eventi_coperti_da_combo > 0 :
            for terno_s_combinazione_popup_dettagliato in items_combinati_popup: migliori_terni_combinati_output_list_popup.append(f"    - {terno_s_combinazione_popup_dettagliato}")
            coperti_popup_val = eventi_coperti_da_combo; totali_popup_val = risultato_miglior_k_terni_popup.get('totale_eventi_spia', len(date_eventi_spia))
            percentuale_popup_val = risultato_miglior_k_terni_popup.get('percentuale_copertura', 0.0)
            migliori_terni_combinati_output_list_popup.append(f"    Copertura combinata: {coperti_popup_val}/{totali_popup_val} Eventi Spia ({percentuale_popup_val:.1f}%).")
        elif not messaggio_specifico: migliori_terni_combinati_output_list_popup.append(f"    Nessuna combinazione significativa trovata o nessun evento coperto.")
    
    if migliori_terni_combinati_output_list_popup: popup_content_list.extend(migliori_terni_combinati_output_list_popup)
    elif 'migliori_terni_copertura_globale' in info_ricerca and info_ricerca.get('migliori_terni_copertura_globale') is not None and not risultato_miglior_k_terni_popup and 'date_trigger_ordinate' in info_ricerca and len(info_ricerca['date_trigger_ordinate']) > 0:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI COMBINAZIONI DI TERNI PER COPERTURA GLOBALE " + "=" * 10)
        popup_content_list.append("  Non è stato possibile trovare combinazioni significative di terni,")
        popup_content_list.append("  oppure non sono stati trovati terni individuali con copertura globale sufficiente per la ricerca.")


    final_popup_text_content = "\n".join(popup_content_list)
    text_area_popup.config(state=tk.NORMAL)
    text_area_popup.delete(1.0, tk.END)
    text_area_popup.insert(tk.END, final_popup_text_content)
    text_area_popup.config(state=tk.DISABLED)

    button_frame_popup_spia = ttk.Frame(popup)
    button_frame_popup_spia.pack(fill=tk.X, pady=(5,10), padx=10, side=tk.BOTTOM)

    def _salva_popup_spia_content_definitiva():
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Salva Riepilogo Analisi Spia"
        )
        if fpath:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(final_popup_text_content)
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
    global risultati_globali, info_ricerca_globale, file_ruote, risultato_text, root
    global start_date_entry, end_date_entry, listbox_ruote_analisi, listbox_ruote_verifica
    global entry_numeri_spia, estrazioni_entry_succ, listbox_ruote_analisi_ant
    global entry_numeri_obiettivo, estrazioni_entry_ant, tipo_spia_var_global, combo_posizione_spia
    global MAX_COLPI_GIOCO

    if not mappa_file_ruote() or not file_ruote:
        messagebox.showerror("Errore Cartella", "Impossibile leggere i file dalla cartella specificata o la cartella non contiene file validi.\nAssicurati di aver selezionato una cartella con 'Sfoglia...'.")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
        risultato_text.insert(tk.END, "Errore: Cartella o file non validi."); risultato_text.config(state=tk.NORMAL)
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
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Errore input date."); risultato_text.config(state=tk.NORMAL)
        return

    messaggi_out,ris_graf_loc = [],[]
    col_num_nomi = [f'Numero{i+1}' for i in range(5)]

    if modalita == "successivi":
        ra_idx,rv_idx = listbox_ruote_analisi.curselection(),listbox_ruote_verifica.curselection()
        if not ra_idx or not rv_idx:
            messagebox.showwarning("Manca Input","Seleziona Ruote Analisi (Spia) e Ruote Verifica (Esiti).")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
        nomi_ra,nomi_rv = [listbox_ruote_analisi.get(i) for i in ra_idx],[listbox_ruote_verifica.get(i) for i in rv_idx]
        tipo_spia_scelto = tipo_spia_var_global.get() if tipo_spia_var_global else "estratto"
        numeri_spia_input_raw = [e.get().strip() for e in entry_numeri_spia]
        numeri_spia_input_validi_zfill = [str(int(n_str)).zfill(2) for n_str in numeri_spia_input_raw if n_str.isdigit() and 1 <= int(n_str) <= 90]
        numeri_spia_da_usare = None; spia_display_str = ""; posizione_spia_selezionata = None
        if tipo_spia_scelto == "estratto":
            if not numeri_spia_input_validi_zfill: messagebox.showwarning("Manca Input","Nessun Numero Spia (Estratto) valido.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            numeri_spia_da_usare = numeri_spia_input_validi_zfill; spia_display_str = ", ".join(numeri_spia_da_usare)
        elif tipo_spia_scelto == "estratto_posizionale":
            if not numeri_spia_input_validi_zfill or not numeri_spia_input_raw[0].strip(): messagebox.showwarning("Manca Input","Inserire il primo Numero Spia per l'analisi posizionale.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            primo_numero_spia = numeri_spia_input_validi_zfill[0]; numeri_spia_da_usare = [primo_numero_spia]; posizione_scelta_str = combo_posizione_spia.get()
            if not posizione_scelta_str or "Qualsiasi" in posizione_scelta_str: tipo_spia_scelto = "estratto"; spia_display_str = primo_numero_spia
            else:
                try: posizione_spia_selezionata = int(posizione_scelta_str.split("a")[0]); assert 1 <= posizione_spia_selezionata <= 5; spia_display_str = f"{primo_numero_spia} in {posizione_spia_selezionata}a pos."
                except: messagebox.showerror("Input Invalido",f"Posizione Spia non valida.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input non valido.");risultato_text.config(state=tk.NORMAL);return
        elif tipo_spia_scelto == "ambo":
            if len(numeri_spia_input_validi_zfill) < 2: messagebox.showwarning("Manca Input","Inserire almeno 2 numeri validi per Ambo Spia.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            numeri_spia_da_usare = tuple(sorted(numeri_spia_input_validi_zfill[:2])); spia_display_str = "-".join(numeri_spia_da_usare)
        else: messagebox.showerror("Errore Interno", f"Tipo spia '{tipo_spia_scelto}' non gestito.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Errore tipo spia.");risultato_text.config(state=tk.NORMAL);return
        messaggi_out.append(f"Tipo Spia Analizzata: {tipo_spia_scelto.upper().replace('_',' ')} ({spia_display_str})")
        try: n_estr=int(estrazioni_entry_succ.get()); assert 1 <= n_estr <= MAX_COLPI_GIOCO
        except: messagebox.showerror("Input Invalido",f"N. Estrazioni (1-{MAX_COLPI_GIOCO}) non valido.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input non valido.");risultato_text.config(state=tk.NORMAL);return
        info_curr={'numeri_spia_input':numeri_spia_da_usare,'tipo_spia_usato': tipo_spia_scelto,'ruote_analisi':nomi_ra,'ruote_verifica':nomi_rv,'n_estrazioni':n_estr,'start_date':start_ts,'end_date':end_ts}
        if posizione_spia_selezionata is not None and tipo_spia_scelto == "estratto_posizionale": info_curr['posizione_spia_input'] = posizione_spia_selezionata
        all_date_trig=set(); messaggi_out.append("\n--- FASE 1: Ricerca Date Uscita Spia ---")
        for nome_ra_loop in nomi_ra:
            df_an=carica_dati(file_ruote.get(nome_ra_loop),start_ts,end_ts)
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
            # MODIFICATO TERMINOLOGIA
            if dates_found_this_ruota: all_date_trig.update(dates_found_this_ruota); messaggi_out.append(f"[{nome_ra_loop}] Trovate {len(dates_found_this_ruota)} date per evento spia {spia_display_str}.")
            else: messaggi_out.append(f"[{nome_ra_loop}] Nessuna uscita spia {spia_display_str}.")

        if not all_date_trig:
            messaggi_out.append(f"\nNESSUNA USCITA SPIA TROVATA PER {spia_display_str}.")
            aggiorna_risultati_globali([],info_curr,modalita=modalita)
            final_output_no_trigger = "\n".join(messaggi_out)
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,final_output_no_trigger); risultato_text.config(state=tk.NORMAL); risultato_text.see("1.0")
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali); return
        date_trig_ord=sorted(list(all_date_trig)); n_trig_tot=len(date_trig_ord)
        # MODIFICATO TERMINOLOGIA
        messaggi_out.append(f"\nFASE 1 OK: {n_trig_tot} date totali per evento spia {spia_display_str}."); info_curr['date_trigger_ordinate']=date_trig_ord

        messaggi_out.append("\n--- FASE 2: Analisi Ruote Verifica ---")
        df_cache_ver = {}
        df_cache_completi_per_ritardo = {}
        num_rv_ok = 0

        for nome_rv_loop in nomi_rv:
            # ... (caricamento df_ver_per_analisi_spia e df_ruota_completa_per_ritardo invariato) ...
            df_ver_per_analisi_spia = df_cache_ver.get(nome_rv_loop)
            if df_ver_per_analisi_spia is None:
                temp_df_analisi = carica_dati(file_ruote.get(nome_rv_loop), start_date=start_ts, end_date=end_ts)
                if temp_df_analisi is not None and not temp_df_analisi.empty:
                    df_ver_per_analisi_spia = temp_df_analisi
                    df_cache_ver[nome_rv_loop] = df_ver_per_analisi_spia
            df_ruota_completa_per_ritardo = df_cache_completi_per_ritardo.get(nome_rv_loop)
            if df_ruota_completa_per_ritardo is None:
                df_ruota_completa_per_ritardo = carica_dati(file_ruote.get(nome_rv_loop), start_date=None, end_date=None)
                if df_ruota_completa_per_ritardo is not None and not df_ruota_completa_per_ritardo.empty:
                    df_cache_completi_per_ritardo[nome_rv_loop] = df_ruota_completa_per_ritardo
            if df_ver_per_analisi_spia is None or df_ver_per_analisi_spia.empty:
                 messaggi_out.append(f"[{nome_rv_loop}] No dati Ver. nel periodo per analisi spia."); continue

            res_ver, err_ver = analizza_ruota_verifica(df_ver_per_analisi_spia, date_trig_ord, n_estr, nome_rv_loop)
            
            if err_ver: messaggi_out.append(f"[{nome_rv_loop}] Errore: {err_ver}"); continue
            if res_ver:
                ris_graf_loc.append((nome_rv_loop, spia_display_str, res_ver))
                num_rv_ok += 1
                # MODIFICATO TERMINOLOGIA
                # La chiave 'totale_trigger' in res_ver viene da analizza_ruota_verifica,
                # se vuoi cambiarla anche lì, dovrai modificare analizza_ruota_verifica
                # e aggiornare il recupero qui: res_ver.get('totale_eventi_spia', 0)
                msg_res_v = f"\n=== Risultati Verifica: {nome_rv_loop} (Base: {res_ver.get('totale_trigger', res_ver.get('totale_eventi_spia', 0))} eventi spia) ==="
                for tipo_s_out in ['estratto', 'ambo', 'terno']:
                    res_s_out = res_ver.get(tipo_s_out)
                    if res_s_out:
                        # MODIFICATO TERMINOLOGIA
                        msg_res_v += f"\n--- {tipo_s_out.capitalize()} Successivi ---\n  Top 10 per Presenza (su {res_ver.get('totale_trigger', res_ver.get('totale_eventi_spia', 0))} casi evento spia):\n"
                        if tipo_s_out in ['estratto', 'ambo', 'terno']: res_s_out.setdefault('ritardi_attuali', {})
                        
                        # Usiamo 'top_data' se disponibile dalla versione aggiornata di analizza_ruota_verifica
                        top_pres_data_list = res_s_out.get('presenza', {}).get('top_data')
                        if top_pres_data_list:
                            for i, data_item in enumerate(top_pres_data_list): # Itera sulla lista di dizionari
                                item_str_out = data_item['item']
                                pres = data_item['valore']
                                perc_p_out = data_item['percentuale']
                                freq_p_out = data_item['altra_stat'] # Frequenza per questo item
                                riga_output_base = f"    {i+1}. {item_str_out}: Pres. {pres} ({perc_p_out:.1f}%) | Freq.Tot: {freq_p_out}"
                                # ... (logica ritardo attuale invariata) ...
                                ritardo_attuale_str = ""
                                df_storico_ruota_rit = df_cache_completi_per_ritardo.get(nome_rv_loop)
                                if df_storico_ruota_rit is not None and not df_storico_ruota_rit.empty:
                                    ritardo_val = "N/D"
                                    if tipo_s_out == 'estratto': ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, item_str_out, "estratto", end_ts)
                                    elif tipo_s_out == 'ambo':
                                        try: ambo_tuple_per_ritardo = tuple(item_str_out.split('-'))
                                        except: ambo_tuple_per_ritardo = None
                                        if ambo_tuple_per_ritardo and len(ambo_tuple_per_ritardo) == 2: ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, ambo_tuple_per_ritardo, "ambo", end_ts)
                                        else: ritardo_val = "N/A (parse)"
                                    elif tipo_s_out == 'terno': 
                                        try: terno_tuple_per_ritardo = tuple(item_str_out.split('-'))
                                        except: terno_tuple_per_ritardo = None
                                        if terno_tuple_per_ritardo and len(terno_tuple_per_ritardo) == 3: ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, terno_tuple_per_ritardo, "terno", end_ts)
                                        else: ritardo_val = "N/A (parse)"
                                    if 'ritardi_attuali' in res_s_out and tipo_s_out in ['estratto', 'ambo', 'terno']:
                                        res_s_out['ritardi_attuali'][item_str_out] = ritardo_val
                                    ritardo_attuale_str = f" | Rit.Att: {ritardo_val}"
                                else:
                                    ritardo_attuale_str = " | Rit.Att: N/D (no data full)"
                                    if tipo_s_out in ['estratto', 'ambo', 'terno'] and 'ritardi_attuali' in res_s_out:
                                        res_s_out['ritardi_attuali'][item_str_out] = "N/D (no data full)"
                                msg_res_v += riga_output_base + ritardo_attuale_str + "\n"
                        elif not res_s_out.get('presenza', {}).get('top', pd.Series(dtype=int)).empty: # Fallback alla vecchia struttura se 'top_data' non c'è
                            for i, (item_key_originale, pres) in enumerate(res_s_out['presenza']['top'].items()):
                                # ... (vecchia logica di recupero, potrebbe dare 0% se non corretta in analizza_ruota_verifica)
                                item_str_out = item_key_originale; perc_p_out = res_s_out['presenza']['percentuali'].get(item_key_originale, 0.0)
                                freq_p_out = res_s_out['presenza']['frequenze'].get(item_key_originale, 0)
                                riga_output_base = f"    {i+1}. {item_str_out}: Pres. {pres} ({perc_p_out:.1f}%) | Freq.Tot: {freq_p_out}"
                                # ... (logica ritardo invariata) ...
                                ritardo_attuale_str = "" # Copia e incolla la logica del ritardo da sopra se necessario
                                df_storico_ruota_rit = df_cache_completi_per_ritardo.get(nome_rv_loop)
                                if df_storico_ruota_rit is not None and not df_storico_ruota_rit.empty:
                                    ritardo_val = "N/D"
                                    if tipo_s_out == 'estratto': ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, item_str_out, "estratto", end_ts)
                                    elif tipo_s_out == 'ambo':
                                        try: ambo_tuple_per_ritardo = tuple(item_str_out.split('-'))
                                        except: ambo_tuple_per_ritardo = None
                                        if ambo_tuple_per_ritardo and len(ambo_tuple_per_ritardo) == 2: ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, ambo_tuple_per_ritardo, "ambo", end_ts)
                                        else: ritardo_val = "N/A (parse)"
                                    elif tipo_s_out == 'terno': 
                                        try: terno_tuple_per_ritardo = tuple(item_str_out.split('-'))
                                        except: terno_tuple_per_ritardo = None
                                        if terno_tuple_per_ritardo and len(terno_tuple_per_ritardo) == 3: ritardo_val = calcola_ritardo_attuale(df_storico_ruota_rit, terno_tuple_per_ritardo, "terno", end_ts)
                                        else: ritardo_val = "N/A (parse)"
                                    if 'ritardi_attuali' in res_s_out and tipo_s_out in ['estratto', 'ambo', 'terno']:
                                        res_s_out['ritardi_attuali'][item_str_out] = ritardo_val
                                    ritardo_attuale_str = f" | Rit.Att: {ritardo_val}"
                                else:
                                    ritardo_attuale_str = " | Rit.Att: N/D (no data full)"
                                    if tipo_s_out in ['estratto', 'ambo', 'terno'] and 'ritardi_attuali' in res_s_out:
                                        res_s_out['ritardi_attuali'][item_str_out] = "N/D (no data full)"
                                msg_res_v += riga_output_base + ritardo_attuale_str + "\n"
                        else: msg_res_v += "    Nessuno.\n"
                        
                        msg_res_v+=f"  Top 10 per Frequenza Totale:\n"
                        top_freq_data_list = res_s_out.get('frequenza', {}).get('top_data')
                        if top_freq_data_list:
                             for i, data_item_f in enumerate(top_freq_data_list):
                                item_str_out_f = data_item_f['item']
                                freq_f = data_item_f['valore']
                                perc_f_out_f = data_item_f['percentuale']
                                pres_f_out_f = data_item_f['altra_stat'] # Presenza per questo item
                                # MODIFICATO TERMINOLOGIA
                                msg_res_v+=f"    {i+1}. {item_str_out_f}: Freq.Tot: {freq_f} ({perc_f_out_f:.1f}%) | Pres. su Eventi Spia: {pres_f_out_f}\n"
                        elif not res_s_out.get('frequenza', {}).get('top', pd.Series(dtype=int)).empty: # Fallback
                            for i,(item,freq) in enumerate(res_s_out['frequenza']['top'].items()):
                                item_str_out = item; perc_f_out=res_s_out['frequenza']['percentuali'].get(item,0.0); pres_f_out=res_s_out['frequenza']['presenze'].get(item,0)
                                # MODIFICATO TERMINOLOGIA
                                msg_res_v+=f"    {i+1}. {item_str_out}: Freq.Tot: {freq} ({perc_f_out:.1f}%) | Pres. su Eventi Spia: {pres_f_out}\n"
                        else: msg_res_v+="    Nessuno.\n"

                        if tipo_s_out == 'ambo':
                            migliori_ambi_log_info = res_s_out.get('migliori_per_copertura_trigger')
                            if migliori_ambi_log_info and migliori_ambi_log_info['items']:
                                # MODIFICATO TERMINOLOGIA
                                msg_res_v += f"  Migliori Ambi per Copertura Eventi Spia (su {migliori_ambi_log_info.get('totale_trigger_spia', n_trig_tot)} totali):\n"
                                for ambo_str_log, count_cop_log in migliori_ambi_log_info['items']:
                                    # MODIFICATO TERMINOLOGIA
                                    perc_cop_log = (count_cop_log / migliori_ambi_log_info.get('totale_trigger_spia', n_trig_tot) * 100) if migliori_ambi_log_info.get('totale_trigger_spia', n_trig_tot) > 0 else 0
                                    msg_res_v += f"    - Ambo {ambo_str_log}: Coperti {count_cop_log} eventi spia ({perc_cop_log:.1f}%)\n"
                            elif res_s_out: msg_res_v += "  Migliori Ambi per Copertura Eventi Spia: Nessuno con copertura significativa.\n"
                    else: msg_res_v += f"\n--- {tipo_s_out.capitalize()} Successivi: Nessun risultato ---\n"
                messaggi_out.append(msg_res_v)
            messaggi_out.append("- " * 20)

        if ris_graf_loc and num_rv_ok > 0:
            # ... (Logica per RISULTATI COMBINATI e COPERTURA GLOBALE AMBI/TERNI invariata,
            # ma assicurati che le stringhe di output qui usino "evento spia" o "caso" dove c'era "trigger")
            # Esempio di modifica per una riga:
            # messaggi_out.append("\n\n=== MIGLIORI AMBI PER COPERTURA GLOBALE DEGLI EVENTI SPIA ===")
            # messaggi_out.append(f"(Su {n_trig_tot} eventi spia totali, ...)
            # Dovrai scorrere questa parte e fare le sostituzioni.
            # ===========================================================================
            # SEZIONE RISULTATI COMBINATI (MODIFICARE TERMINOLOGIA "trigger" se presente)
            # ===========================================================================
            messaggi_out.append("\n\n=== RISULTATI COMBINATI (PER PUNTEGGIO - TUTTE RUOTE VERIFICA) ===")
            info_curr['statistiche_combinate_dettagliate'] = {}; info_curr.setdefault('ritardi_attuali_combinati', {})
            top_comb_ver = {'estratto': [], 'ambo': [], 'terno': []}; peso_pres, peso_freq = 0.6, 0.4
            for tipo_comb in ['estratto', 'ambo', 'terno']:
                messaggi_out.append(f"\n--- Combinati: {tipo_comb.upper()} Successivi (per Punteggio) ---")
                comb_pres_dict, comb_freq_dict, has_data_comb = {}, {}, False
                for _, _, res_comb_loop in ris_graf_loc:
                    if res_comb_loop and res_comb_loop.get(tipo_comb):
                        has_data_comb = True; current_presenze_raw = res_comb_loop[tipo_comb].get('full_presenze', pd.Series(dtype=int)); current_frequenze_raw = res_comb_loop[tipo_comb].get('full_frequenze', pd.Series(dtype=int))
                        for item_c_raw_pres, count_c_pres in current_presenze_raw.items():
                            key_pres = tuple(item_c_raw_pres.split('-')) if isinstance(item_c_raw_pres, str) and '-' in item_c_raw_pres and tipo_comb in ['ambo', 'terno'] else item_c_raw_pres
                            comb_pres_dict[key_pres] = comb_pres_dict.get(key_pres, 0) + count_c_pres
                        for item_c_raw_freq, count_c_freq in current_frequenze_raw.items():
                            key_freq = tuple(item_c_raw_freq.split('-')) if isinstance(item_c_raw_freq, str) and '-' in item_c_raw_freq and tipo_comb in ['ambo', 'terno'] else item_c_raw_freq
                            comb_freq_dict[key_freq] = comb_freq_dict.get(key_freq, 0) + count_c_freq
                if not has_data_comb: messaggi_out.append(f"    Nessun risultato combinato per {tipo_comb}.\n"); info_curr['ritardi_attuali_combinati'].setdefault(tipo_comb, {}); continue
                comb_pres_s_orig_keys = pd.Series(comb_pres_dict, dtype=int); comb_freq_s_orig_keys = pd.Series(comb_freq_dict, dtype=int)
                all_items_idx_comb_orig_keys = comb_pres_s_orig_keys.index.union(comb_freq_s_orig_keys.index)
                def get_sortable_key_comb(k): return tuple(map(str, k)) if isinstance(k, tuple) else str(k)
                sortable_index_list_comb = sorted(list(all_items_idx_comb_orig_keys), key=get_sortable_key_comb)
                ordered_index_comb = pd.Index(sortable_index_list_comb)
                comb_pres_s = comb_pres_s_orig_keys.reindex(ordered_index_comb, fill_value=0); comb_freq_s = comb_freq_s_orig_keys.reindex(ordered_index_comb, fill_value=0)
                # MODIFICATO TERMINOLOGIA: n_trig_tot rappresenta il numero di eventi spia
                tot_pres_ops_comb = n_trig_tot * num_rv_ok ; comb_perc_pres_s = (comb_pres_s / tot_pres_ops_comb * 100).round(2) if tot_pres_ops_comb > 0 else pd.Series(0.0, index=comb_pres_s.index, dtype=float)
                max_freq_comb = comb_freq_s.max() if not comb_freq_s.empty else 0; comb_freq_norm_s = (comb_freq_s / max_freq_comb * 100).round(2) if max_freq_comb > 0 else pd.Series(0.0, index=comb_freq_s.index, dtype=float)
                punt_comb_s = ((peso_pres * comb_perc_pres_s) + (peso_freq * comb_freq_norm_s)).round(2).sort_values(ascending=False)
                top_punt_comb = punt_comb_s.head(10); info_curr['ritardi_attuali_combinati'].setdefault(tipo_comb, {})
                if not top_punt_comb.empty:
                    top_comb_ver[tipo_comb] = top_punt_comb.index.tolist()[:10]; stat_dett_comb = []
                    messaggi_out.append(f"  Top 10 Combinati per Punteggio:\n")
                    for i, (item_comb_key_orig, score_comb_loop) in enumerate(top_punt_comb.items()):
                        item_str_formatted_log = format_ambo_terno(item_comb_key_orig); min_ritardo_comb = float('inf'); ritardo_valido_trovato_log = False
                        if tipo_comb in ['estratto', 'ambo', 'terno']:
                            for nome_rv_rit_calc_log in nomi_rv:
                                df_ruota_storico_log = df_cache_completi_per_ritardo.get(nome_rv_rit_calc_log)
                                if df_ruota_storico_log is not None and not df_ruota_storico_log.empty:
                                    current_rit_val_log = "N/D"
                                    if tipo_comb == 'estratto': current_rit_val_log = calcola_ritardo_attuale(df_ruota_storico_log, item_comb_key_orig, "estratto", end_ts)
                                    elif tipo_comb == 'ambo': current_rit_val_log = calcola_ritardo_attuale(df_ruota_storico_log, item_comb_key_orig, "ambo", end_ts)
                                    elif tipo_comb == 'terno': current_rit_val_log = calcola_ritardo_attuale(df_ruota_storico_log, item_comb_key_orig, "terno", end_ts)
                                    if isinstance(current_rit_val_log, (int, float)): min_ritardo_comb = min(min_ritardo_comb, current_rit_val_log); ritardo_valido_trovato_log = True
                        ritardo_display_str_log = ""; ritardo_da_memorizzare_log = "N/A"
                        if ritardo_valido_trovato_log: ritardo_display_str_log = f" | Rit.Min.Att: {int(min_ritardo_comb)}"; ritardo_da_memorizzare_log = int(min_ritardo_comb)
                        elif tipo_comb in ['estratto', 'ambo', 'terno']: ritardo_display_str_log = " | Rit.Min.Att: N/D"; ritardo_da_memorizzare_log = "N/D"
                        if tipo_comb in ['estratto', 'ambo', 'terno']:
                            info_curr['ritardi_attuali_combinati'][tipo_comb][item_str_formatted_log] = ritardo_da_memorizzare_log
                        pres_avg = comb_perc_pres_s.get(item_comb_key_orig, 0.0); freq_tot = comb_freq_s.get(item_comb_key_orig, 0)
                        messaggi_out.append(f"    {i+1}. {item_str_formatted_log}: Punt={score_comb_loop:.2f} (PresAvg:{pres_avg:.1f}%, FreqTot:{freq_tot}){ritardo_display_str_log}\n")
                        dettaglio_comb = { "item": item_str_formatted_log, "punteggio": score_comb_loop, "presenza_media_perc": pres_avg, "frequenza_totale": freq_tot }
                        if tipo_comb in ['estratto', 'ambo', 'terno']:
                            dettaglio_comb["ritardo_min_attuale"] = ritardo_da_memorizzare_log
                        stat_dett_comb.append(dettaglio_comb)
                    info_curr.setdefault('statistiche_combinate_dettagliate', {})[tipo_comb] = stat_dett_comb
                else: messaggi_out.append("    Nessuno.\n")
            info_curr['top_combinati'] = {k: [format_ambo_terno(item) for item in v_list] for k, v_list in top_comb_ver.items()}
        elif num_rv_ok == 0: messaggi_out.append("\nNessuna Ruota Verifica valida con risultati.")

        # ===========================================================================
        # SEZIONI ESTRATTI, AMBI, TERNI GLOBALI (MODIFICARE TERMINOLOGIA "evento spia")
        # ===========================================================================
        # --- ESTRATTI GLOBALI ---
        if ris_graf_loc and num_rv_ok > 0 and n_trig_tot > 0:
            messaggi_out.append("\n\n=== MIGLIORI ESTRATTI INDIVIDUALI PER COPERTURA GLOBALE DEGLI EVENTI SPIA ===")
            messaggi_out.append(f"(Su {n_trig_tot} eventi spia totali, considerando uscite su QUALSIASI ruota di verifica entro {n_estr} colpi)")
            # ... (continua la logica degli estratti, assicurati che le stampe usino "eventi spia") ...
            estratti_copertura_globale_eventi = {} 
            for data_trigger_evento_estratto in date_trig_ord:
                estratti_usciti_per_questo_trigger_globale = set()
                for nome_rv_check_glob_estratto in nomi_rv:
                    df_ver_loop_estratto = df_cache_completi_per_ritardo.get(nome_rv_check_glob_estratto)
                    if df_ver_loop_estratto is None or df_ver_loop_estratto.empty: continue
                    date_series_ver_loop_estratto = df_ver_loop_estratto['Data']
                    try: start_index_loop_estratto = date_series_ver_loop_estratto.searchsorted(data_trigger_evento_estratto, side='right')
                    except Exception: continue
                    if start_index_loop_estratto >= len(date_series_ver_loop_estratto): continue
                    df_successive_loop_estratto = df_ver_loop_estratto.iloc[start_index_loop_estratto : start_index_loop_estratto + n_estr]
                    if not df_successive_loop_estratto.empty:
                        for _, row_loop_estratto in df_successive_loop_estratto.iterrows():
                            numeri_riga_loop_estratto_str = {str(row_loop_estratto[col]).zfill(2) for col in col_num_nomi if pd.notna(row_loop_estratto[col])}
                            estratti_usciti_per_questo_trigger_globale.update(numeri_riga_loop_estratto_str)
                for estratto_coperto_glob in estratti_usciti_per_questo_trigger_globale:
                    if estratto_coperto_glob not in estratti_copertura_globale_eventi: estratti_copertura_globale_eventi[estratto_coperto_glob] = set()
                    estratti_copertura_globale_eventi[estratto_coperto_glob].add(data_trigger_evento_estratto)
            conteggio_copertura_estratti_globale = Counter({estratto_glob: len(date_coperte_glob) for estratto_glob, date_coperte_glob in estratti_copertura_globale_eventi.items()})
            if not conteggio_copertura_estratti_globale:
                messaggi_out.append("    Nessun estratto ha coperto eventi spia (considerando tutte le ruote di verifica).")
                info_curr['migliori_estratti_copertura_globale'] = []
            else:
                migliori_estratti_globali_raw = sorted(conteggio_copertura_estratti_globale.items(), key=lambda item: (item[1], item[0]), reverse=True)
                num_top_estratti_globali_display = min(len(migliori_estratti_globali_raw), 10)
                info_curr['migliori_estratti_copertura_globale'] = []
                if num_top_estratti_globali_display > 0 :
                    messaggi_out.append(f"  Top {num_top_estratti_globali_display} estratti individuali (per copertura eventi spia):")
                    for i in range(num_top_estratti_globali_display):
                        estratto_glob_str, count_glob_est = migliori_estratti_globali_raw[i]
                        perc_glob_est = (count_glob_est / n_trig_tot * 100) if n_trig_tot > 0 else 0.0
                        min_rit_estratto_glob = float('inf'); rit_valido_estratto_glob_trovato = False
                        for nome_rv_rit_calc_glob_est in nomi_rv:
                            df_ruota_storico_glob_est = df_cache_completi_per_ritardo.get(nome_rv_rit_calc_glob_est)
                            if df_ruota_storico_glob_est is not None and not df_ruota_storico_glob_est.empty:
                                current_rit_val_glob_est = calcola_ritardo_attuale(df_ruota_storico_glob_est, estratto_glob_str, "estratto", end_ts)
                                if isinstance(current_rit_val_glob_est, (int, float)): min_rit_estratto_glob = min(min_rit_estratto_glob, current_rit_val_glob_est); rit_valido_estratto_glob_trovato = True
                        rit_display_est_glob_str = ""; rit_da_mem_est_glob = "N/A"
                        if rit_valido_estratto_glob_trovato: rit_display_est_glob_str = f" | Rit.Min.Att: {int(min_rit_estratto_glob)}"; rit_da_mem_est_glob = int(min_rit_estratto_glob)
                        else: rit_display_est_glob_str = " | Rit.Min.Att: N/D"; rit_da_mem_est_glob = "N/D"
                        messaggi_out.append(f"    {i+1}. Estratto {estratto_glob_str}: Coperti {count_glob_est} su {n_trig_tot} eventi spia ({perc_glob_est:.1f}%){rit_display_est_glob_str}")
                        info_curr['migliori_estratti_copertura_globale'].append({ "estratto": estratto_glob_str, "coperti": count_glob_est, "totali": n_trig_tot, "percentuale": perc_glob_est, "ritardo_min_attuale": rit_da_mem_est_glob })
                else: messaggi_out.append("    Nessun estratto individuale con copertura significativa trovato.")
                info_curr['combinazione_ottimale_estratti_100'] = None; info_curr['migliore_combinazione_parziale_estratti'] = None
                if info_curr['migliori_estratti_copertura_globale'] and n_trig_tot > 0:
                    messaggi_out.append(f"\n  RICERCA MIGLIORE COMBINAZIONE ESTRATTI (MAX 3) PER COPERTURA ({n_trig_tot} eventi spia):")
                    top_estratti_tuple_per_combinazioni = [item[0] for item in migliori_estratti_globali_raw[:20]]
                    ritardi_min_attuali_estratti_individuali = {e_info['estratto']: e_info['ritardo_min_attuale'] for e_info in info_curr['migliori_estratti_copertura_globale']}
                    soluzione_100_trovata_estr = False; max_copertura_parziale_estr = -1
                    for k_estratti_comb in range(1, 4):
                        if len(top_estratti_tuple_per_combinazioni) < k_estratti_comb: break 
                        miglior_combinazione_k_attuale_estr = None; copertura_per_miglior_combinazione_k_attuale_estr = -1
                        for combo_estr_corrente_tuple in itertools.combinations(top_estratti_tuple_per_combinazioni, k_estratti_comb):
                            combo_estr_ordinata_tuple = tuple(sorted(list(combo_estr_corrente_tuple)))
                            date_coperte_da_questa_combo = set().union(*(estratti_copertura_globale_eventi.get(e_in_combo, set()) for e_in_combo in combo_estr_ordinata_tuple))
                            copertura_di_questa_combo = len(date_coperte_da_questa_combo)
                            if copertura_di_questa_combo > copertura_per_miglior_combinazione_k_attuale_estr:
                                copertura_per_miglior_combinazione_k_attuale_estr = copertura_di_questa_combo; miglior_combinazione_k_attuale_estr = combo_estr_ordinata_tuple
                            elif copertura_di_questa_combo == copertura_per_miglior_combinazione_k_attuale_estr and (miglior_combinazione_k_attuale_estr is None or combo_estr_ordinata_tuple < miglior_combinazione_k_attuale_estr):
                                miglior_combinazione_k_attuale_estr = combo_estr_ordinata_tuple
                        if miglior_combinazione_k_attuale_estr:
                            estratti_dettagli_list_k = [f"{e_str_k}{(' [Rit.Min.Att: ' + str(int(r)) + ']' if isinstance(r := ritardi_min_attuali_estratti_individuali.get(e_str_k), int) else (' [Rit.Min.Att: N/D]' if r == 'N/D' else ''))}" for e_str_k in sorted(list(miglior_combinazione_k_attuale_estr))]
                            percentuale_k_estr = (copertura_per_miglior_combinazione_k_attuale_estr / n_trig_tot * 100) if n_trig_tot > 0 else 0.0
                            info_combinazione_k = {"estratti_dettagli": estratti_dettagli_list_k, "estratti": list(miglior_combinazione_k_attuale_estr), "coperti": copertura_per_miglior_combinazione_k_attuale_estr, "totali": n_trig_tot, "percentuale": percentuale_k_estr, "num_estratti_nella_combinazione": k_estratti_comb}
                            if copertura_per_miglior_combinazione_k_attuale_estr == n_trig_tot:
                                info_curr['combinazione_ottimale_estratti_100'] = info_combinazione_k; soluzione_100_trovata_estr = True
                                messaggi_out.append(f"  - Combinazione di {k_estratti_comb} estratti copre il 100% ({n_trig_tot}/{n_trig_tot} eventi spia):"); _=[messaggi_out.append(f"    - {e_dett_100}") for e_dett_100 in estratti_dettagli_list_k]; break
                            if copertura_per_miglior_combinazione_k_attuale_estr > max_copertura_parziale_estr:
                                max_copertura_parziale_estr = copertura_per_miglior_combinazione_k_attuale_estr; info_curr['migliore_combinazione_parziale_estratti'] = info_combinazione_k
                    if not soluzione_100_trovata_estr and info_curr['migliore_combinazione_parziale_estratti']:
                        parziale_estr_info = info_curr['migliore_combinazione_parziale_estratti']; num_e_parz = parziale_estr_info['num_estratti_nella_combinazione']
                        messaggi_out.append(f"  - Non è stata trovata una combinazione (max 3 estratti) per la copertura del 100%.\n    La migliore copertura parziale trovata con {num_e_parz} estratti è di {parziale_estr_info['coperti']}/{parziale_estr_info['totali']} eventi spia ({parziale_estr_info['percentuale']:.1f}%):"); _=[messaggi_out.append(f"      - {e_dett_p_log}") for e_dett_p_log in parziale_estr_info['estratti_dettagli']]
        # --- AMBI GLOBALI ---
        if ris_graf_loc and num_rv_ok > 0 and n_trig_tot > 0:
            messaggi_out.append("\n\n=== MIGLIORI AMBI PER COPERTURA GLOBALE DEGLI EVENTI SPIA ===")
            messaggi_out.append(f"(Su {n_trig_tot} eventi spia totali, considerando uscite su QUALSIASI ruota di verifica entro {n_estr} colpi)")
            # ... (continua la logica degli ambi, assicurati che le stampe usino "eventi spia") ...
            ambi_copertura_globale_eventi = {} # Inizio logica ambi
            for data_trigger_evento in date_trig_ord: 
                ambi_usciti_per_questo_trigger_globale = set()
                for nome_rv_check_glob in nomi_rv:
                    df_ver_loop_glob = df_cache_completi_per_ritardo.get(nome_rv_check_glob)
                    if df_ver_loop_glob is None or df_ver_loop_glob.empty: continue
                    date_series_ver_loop_glob = df_ver_loop_glob['Data']
                    try: start_index_loop_glob = date_series_ver_loop_glob.searchsorted(data_trigger_evento, side='right')
                    except Exception: continue
                    if start_index_loop_glob >= len(date_series_ver_loop_glob): continue
                    df_successive_loop_glob = df_ver_loop_glob.iloc[start_index_loop_glob : start_index_loop_glob + n_estr]
                    if not df_successive_loop_glob.empty:
                        for _, row_loop_glob in df_successive_loop_glob.iterrows():
                            numeri_riga_loop_glob_str = [str(row_loop_glob[col]).zfill(2) for col in col_num_nomi if pd.notna(row_loop_glob[col])]
                            numeri_riga_loop_glob = sorted(numeri_riga_loop_glob_str)
                            if len(numeri_riga_loop_glob) >= 2:
                                for ambo_tuple_loop in itertools.combinations(numeri_riga_loop_glob, 2): ambi_usciti_per_questo_trigger_globale.add(ambo_tuple_loop)
                for ambo_coperto_glob in ambi_usciti_per_questo_trigger_globale:
                    if ambo_coperto_glob not in ambi_copertura_globale_eventi: ambi_copertura_globale_eventi[ambo_coperto_glob] = set()
                    ambi_copertura_globale_eventi[ambo_coperto_glob].add(data_trigger_evento)
            conteggio_copertura_globale = Counter({ambo_glob: len(date_coperte_glob) for ambo_glob, date_coperte_glob in ambi_copertura_globale_eventi.items()})
            if not conteggio_copertura_globale:
                messaggi_out.append("    Nessun ambo ha coperto eventi spia (considerando tutte le ruote di verifica).")
                info_curr['migliori_ambi_copertura_globale'] = []
            else:
                migliori_ambi_globali_raw = sorted(conteggio_copertura_globale.items(), key=lambda item: (item[1], item[0][0], item[0][1]), reverse=True)
                num_top_ambi_globali_display = min(len(migliori_ambi_globali_raw), 10); info_curr['migliori_ambi_copertura_globale'] = []
                if num_top_ambi_globali_display > 0 :
                    messaggi_out.append(f"  Top {num_top_ambi_globali_display} ambi individuali (per copertura eventi spia):")
                    for i in range(num_top_ambi_globali_display):
                        ambo_glob_tuple, count_glob = migliori_ambi_globali_raw[i]; ambo_glob_str = format_ambo_terno(ambo_glob_tuple)
                        perc_glob = (count_glob / n_trig_tot * 100) if n_trig_tot > 0 else 0
                        min_rit_ambo_glob = float('inf'); rit_valido_ambo_glob_trovato = False
                        for nome_rv_rit_calc_glob in nomi_rv:
                            df_ruota_storico_glob = df_cache_completi_per_ritardo.get(nome_rv_rit_calc_glob)
                            if df_ruota_storico_glob is not None and not df_ruota_storico_glob.empty:
                                current_rit_val_glob = calcola_ritardo_attuale(df_ruota_storico_glob, ambo_glob_tuple, "ambo", end_ts)
                                if isinstance(current_rit_val_glob, (int, float)): min_rit_ambo_glob = min(min_rit_ambo_glob, current_rit_val_glob); rit_valido_ambo_glob_trovato = True
                        rit_display_ambo_glob_str = ""; rit_da_mem_ambo_glob = "N/A"
                        if rit_valido_ambo_glob_trovato: rit_display_ambo_glob_str = f" | Rit.Min.Att: {int(min_rit_ambo_glob)}"; rit_da_mem_ambo_glob = int(min_rit_ambo_glob)
                        else: rit_display_ambo_glob_str = " | Rit.Min.Att: N/D"; rit_da_mem_ambo_glob = "N/D"
                        messaggi_out.append(f"    {i+1}. Ambo {ambo_glob_str}: Coperti {count_glob} su {n_trig_tot} eventi spia ({perc_glob:.1f}%){rit_display_ambo_glob_str}")
                        info_curr['migliori_ambi_copertura_globale'].append({ "ambo": ambo_glob_str, "coperti": count_glob, "totali": n_trig_tot, "percentuale": perc_glob, "ritardo_min_attuale": rit_da_mem_ambo_glob })
                    info_curr['combinazione_ottimale_copertura_100'] = None; info_curr['migliore_combinazione_parziale'] = None
                    if len(info_curr['migliori_ambi_copertura_globale']) > 0 and n_trig_tot > 0:
                        messaggi_out.append(f"\n  RICERCA COMBINAZIONE AMBI PER COPERTURA TOTALE ({n_trig_tot} eventi spia):")
                        top_ambi_tuple_per_combinazioni = [info[0] for info in migliori_ambi_globali_raw[:7]]; soluzione_trovata = False
                        if top_ambi_tuple_per_combinazioni:
                            ambo_top1_tuple_raw_comb = top_ambi_tuple_per_combinazioni[0]; date_coperte_top1_comb = ambi_copertura_globale_eventi.get(ambo_top1_tuple_raw_comb, set())
                            ambo_top1_info_det_comb = next((item for item in info_curr['migliori_ambi_copertura_globale'] if item["ambo"] == format_ambo_terno(ambo_top1_tuple_raw_comb)), None)
                            ambo_top1_str_det_comb = format_ambo_terno(ambo_top1_tuple_raw_comb)
                            if ambo_top1_info_det_comb and ambo_top1_info_det_comb['ritardo_min_attuale'] not in ['N/A', 'N/D', None]: ambo_top1_str_det_comb += f" [Rit.Min.Att: {ambo_top1_info_det_comb['ritardo_min_attuale']}]"
                            elif ambo_top1_info_det_comb and ambo_top1_info_det_comb['ritardo_min_attuale'] == 'N/D': ambo_top1_str_det_comb += " [Rit.Min.Att: N/D]"
                            if len(date_coperte_top1_comb) == n_trig_tot:
                                messaggi_out.append(f"  - L'ambo singolo '{ambo_top1_str_det_comb}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi spia).") # MODIFICATO TERMINOLOGIA
                                info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": [ambo_top1_str_det_comb], "ambi": [format_ambo_terno(ambo_top1_tuple_raw_comb)], "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}; soluzione_trovata = True
                        if not soluzione_trovata and len(top_ambi_tuple_per_combinazioni) >= 2:
                            # ... (continua logica combinazione ambi, assicurati di cambiare "eventi" in "eventi spia" nelle stampe) ...
                            miglior_coppia_copertura = 0; miglior_coppia_ambi_dettagli_list = []; miglior_coppia_ambi_raw_list = []
                            for combo_2_tuple_ambi in itertools.combinations(top_ambi_tuple_per_combinazioni, 2):
                                amboA_tuple_comb2, amboB_tuple_comb2 = combo_2_tuple_ambi[0], combo_2_tuple_ambi[1]; date_A_comb2 = ambi_copertura_globale_eventi.get(amboA_tuple_comb2, set()); date_B_comb2 = ambi_copertura_globale_eventi.get(amboB_tuple_comb2, set()); coperte_da_coppia_comb2 = date_A_comb2.union(date_B_comb2)
                                current_ambi_dettagli_list_comb2 = []
                                for ambo_t_comb2_loop in [amboA_tuple_comb2, amboB_tuple_comb2]:
                                    info_det_comb2 = next((item for item in info_curr['migliori_ambi_copertura_globale'] if item["ambo"] == format_ambo_terno(ambo_t_comb2_loop)), None); str_det_comb2 = format_ambo_terno(ambo_t_comb2_loop)
                                    if info_det_comb2 and info_det_comb2['ritardo_min_attuale'] not in ['N/A', 'N/D', None]: str_det_comb2 += f" [Rit.Min.Att: {info_det_comb2['ritardo_min_attuale']}]"
                                    elif info_det_comb2 and info_det_comb2['ritardo_min_attuale'] == 'N/D': str_det_comb2 += " [Rit.Min.Att: N/D]"
                                    current_ambi_dettagli_list_comb2.append(str_det_comb2)
                                current_ambi_dettagli_list_comb2.sort()
                                if len(coperte_da_coppia_comb2) == n_trig_tot:
                                    miglior_coppia_ambi_dettagli_list = current_ambi_dettagli_list_comb2; miglior_coppia_ambi_raw_list = [format_ambo_terno(amboA_tuple_comb2), format_ambo_terno(amboB_tuple_comb2)]
                                    messaggi_out.append(f"  - La coppia '{miglior_coppia_ambi_dettagli_list[0]}' e '{miglior_coppia_ambi_dettagli_list[1]}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi spia).")
                                    info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": miglior_coppia_ambi_dettagli_list, "ambi": sorted(miglior_coppia_ambi_raw_list), "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}; soluzione_trovata = True; break
                                if len(coperte_da_coppia_comb2) > miglior_coppia_copertura:
                                    miglior_coppia_copertura = len(coperte_da_coppia_comb2); miglior_coppia_ambi_dettagli_list = current_ambi_dettagli_list_comb2; miglior_coppia_ambi_raw_list = [format_ambo_terno(amboA_tuple_comb2), format_ambo_terno(amboB_tuple_comb2)]
                            if not soluzione_trovata and miglior_coppia_ambi_dettagli_list: info_curr['migliore_combinazione_parziale'] = {"ambi_dettagli": miglior_coppia_ambi_dettagli_list, "ambi": sorted(miglior_coppia_ambi_raw_list), "coperti": miglior_coppia_copertura, "totali": n_trig_tot, "percentuale": (miglior_coppia_copertura/n_trig_tot*100) if n_trig_tot > 0 else 0}
                        if not soluzione_trovata and len(top_ambi_tuple_per_combinazioni) >= 3:
                            miglior_terzina_copertura = 0; miglior_terzina_ambi_dettagli_list = []; miglior_terzina_ambi_raw_list = []
                            for combo_3_tuple_ambi in itertools.combinations(top_ambi_tuple_per_combinazioni, 3):
                                amboA_t3, amboB_t3, amboC_t3 = combo_3_tuple_ambi[0], combo_3_tuple_ambi[1], combo_3_tuple_ambi[2]; date_A_t3 = ambi_copertura_globale_eventi.get(amboA_t3, set()); date_B_t3 = ambi_copertura_globale_eventi.get(amboB_t3, set()); date_C_t3 = ambi_copertura_globale_eventi.get(amboC_t3, set()); coperte_da_terzina_t3 = date_A_t3.union(date_B_t3).union(date_C_t3)
                                current_ambi_dettagli_list_comb3 = []
                                for ambo_t_comb3_loop in [amboA_t3, amboB_t3, amboC_t3]:
                                    info_det_comb3 = next((item for item in info_curr['migliori_ambi_copertura_globale'] if item["ambo"] == format_ambo_terno(ambo_t_comb3_loop)), None); str_det_comb3 = format_ambo_terno(ambo_t_comb3_loop)
                                    if info_det_comb3 and info_det_comb3['ritardo_min_attuale'] not in ['N/A', 'N/D', None]: str_det_comb3 += f" [Rit.Min.Att: {info_det_comb3['ritardo_min_attuale']}]"
                                    elif info_det_comb3 and info_det_comb3['ritardo_min_attuale'] == 'N/D': str_det_comb3 += " [Rit.Min.Att: N/D]"
                                    current_ambi_dettagli_list_comb3.append(str_det_comb3)
                                current_ambi_dettagli_list_comb3.sort()
                                if len(coperte_da_terzina_t3) == n_trig_tot:
                                    miglior_terzina_ambi_dettagli_list = current_ambi_dettagli_list_comb3; miglior_terzina_ambi_raw_list = [format_ambo_terno(amboA_t3), format_ambo_terno(amboB_t3), format_ambo_terno(amboC_t3)]
                                    messaggi_out.append(f"  - La terzina '{miglior_terzina_ambi_dettagli_list[0]}', '{miglior_terzina_ambi_dettagli_list[1]}' e '{miglior_terzina_ambi_dettagli_list[2]}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi spia).")
                                    info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": miglior_terzina_ambi_dettagli_list, "ambi": sorted(miglior_terzina_ambi_raw_list), "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}; soluzione_trovata = True; break
                                if len(coperte_da_terzina_t3) > miglior_terzina_copertura:
                                    miglior_terzina_copertura = len(coperte_da_terzina_t3); miglior_terzina_ambi_dettagli_list = current_ambi_dettagli_list_comb3; miglior_terzina_ambi_raw_list = [format_ambo_terno(amboA_t3), format_ambo_terno(amboB_t3), format_ambo_terno(amboC_t3)]
                            if not soluzione_trovata and miglior_terzina_ambi_dettagli_list and (not info_curr.get('migliore_combinazione_parziale') or miglior_terzina_copertura > info_curr['migliore_combinazione_parziale']['coperti']):
                                info_curr['migliore_combinazione_parziale'] = {"ambi_dettagli": miglior_terzina_ambi_dettagli_list, "ambi": sorted(miglior_terzina_ambi_raw_list), "coperti": miglior_terzina_copertura, "totali": n_trig_tot, "percentuale": (miglior_terzina_copertura/n_trig_tot*100) if n_trig_tot > 0 else 0}
                        if not soluzione_trovata:
                            messaggi_out.append("  - Non è stata trovata una combinazione di 1, 2 o 3 ambi (dai top considerati) per la copertura del 100%.")
                            if info_curr['migliore_combinazione_parziale']:
                                parziale = info_curr['migliore_combinazione_parziale']; ambi_da_mostrare_log_parziale = parziale.get("ambi_dettagli", parziale["ambi"])
                                messaggi_out.append(f"    La migliore copertura parziale trovata con {len(parziale['ambi'])} ambo/i ({', '.join(ambi_da_mostrare_log_parziale)}) è di {parziale['coperti']}/{parziale['totali']} eventi spia ({parziale['percentuale']:.1f}%).") # MODIFICATO TERMINOLOGIA
                else: messaggi_out.append("    Nessun ambo ha coperto eventi spia in modo significativo (considerando tutte le ruote di verifica).")
        # --- TERNI GLOBALI ---
        if ris_graf_loc and num_rv_ok > 0 and n_trig_tot > 0:
            messaggi_out.append("\n\n=== MIGLIORI TERNI PER COPERTURA GLOBALE DEGLI EVENTI SPIA (INDIVIDUALE) ===")
            messaggi_out.append(f"(Su {n_trig_tot} eventi spia totali, uscite su QUALSIASI ruota di verifica entro {n_estr} colpi)")
            # ... (continua la logica dei terni, assicurati che le stampe usino "eventi spia") ...
            # Esempio:
            # messaggi_out.append(f"    {i+1}. Terno {terno_glob_str}: Coperti {count_glob_terno} su {n_trig_tot} eventi spia ({perc_glob_terno:.1f}%){rit_individual_display}")
            # ... e per le combinazioni:
            # messaggi_out.append(f"    Copertura combinata: {eventi_cop_log} su {n_trig_tot} eventi spia ({perc_cop_log:.1f}%)")
            terni_copertura_globale_eventi = {} # Inizio logica terni
            for data_trigger_evento_terni in date_trig_ord:
                terni_usciti_per_questo_trigger_glob_terni = set()
                for nome_rv_check_glob_terni in nomi_rv:
                    df_ver_loop_terni = df_cache_completi_per_ritardo.get(nome_rv_check_glob_terni)
                    if df_ver_loop_terni is None or df_ver_loop_terni.empty: continue
                    date_series_ver_loop_terni = df_ver_loop_terni['Data']
                    try: start_index_loop_terni = date_series_ver_loop_terni.searchsorted(data_trigger_evento_terni, side='right')
                    except Exception: continue
                    if start_index_loop_terni >= len(date_series_ver_loop_terni): continue
                    df_successive_loop_terni = df_ver_loop_terni.iloc[start_index_loop_terni : start_index_loop_terni + n_estr]
                    if not df_successive_loop_terni.empty:
                        for _, row_loop_terni in df_successive_loop_terni.iterrows():
                            numeri_riga_loop_terni_str = [str(row_loop_terni[col]).zfill(2) for col in col_num_nomi if pd.notna(row_loop_terni[col])]
                            numeri_riga_loop_terni = sorted(numeri_riga_loop_terni_str)
                            if len(numeri_riga_loop_terni) >= 3:
                                for terno_tuple_loop in itertools.combinations(numeri_riga_loop_terni, 3): terni_usciti_per_questo_trigger_glob_terni.add(terno_tuple_loop)
                for terno_coperto_glob in terni_usciti_per_questo_trigger_glob_terni:
                    if terno_coperto_glob not in terni_copertura_globale_eventi: terni_copertura_globale_eventi[terno_coperto_glob] = set()
                    terni_copertura_globale_eventi[terno_coperto_glob].add(data_trigger_evento_terni)
            conteggio_copertura_terni_globale = Counter({ terno_glob: len(date_coperte_glob) for terno_glob, date_coperte_glob in terni_copertura_globale_eventi.items() })
            NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI = 25
            if not conteggio_copertura_terni_globale:
                messaggi_out.append("    Nessun terno ha coperto eventi spia (considerando tutte le ruote di verifica).")
                info_curr['migliori_terni_copertura_globale'] = []; info_curr['config_combinazioni_terni'] = {'num_top_terni_considerati': NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI}
                for k_val_terni_err in [3,4,5]: info_curr[f'migliore_combinazione_{k_val_terni_err}_terni'] = {"messaggio": "Nessun terno individuale trovato per formare combinazioni."}
                messaggi_out.append("\n    Impossibile cercare combinazioni di terni perché non sono stati trovati terni con copertura individuale.")
            else:
                migliori_terni_globali_raw = sorted( conteggio_copertura_terni_globale.items(), key=lambda item: (item[1], item[0][0], item[0][1], item[0][2]), reverse=True )
                num_top_terni_individuali_display = min(len(migliori_terni_globali_raw), 10); info_curr['migliori_terni_copertura_globale'] = []
                ritardi_min_attuali_terni_individuali = {}
                items_terni_per_ritardo_calc = [item[0] for item in migliori_terni_globali_raw[:NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI]]
                for terno_tuple_rit_calc in items_terni_per_ritardo_calc:
                    min_rit_terno_glob = float('inf'); rit_valido_terno_glob_trovato = False
                    for nome_rv_rit_calc_glob_terno in nomi_rv:
                        df_ruota_storico_glob_terno = df_cache_completi_per_ritardo.get(nome_rv_rit_calc_glob_terno)
                        if df_ruota_storico_glob_terno is not None and not df_ruota_storico_glob_terno.empty:
                            current_rit_val_glob_terno = calcola_ritardo_attuale(df_ruota_storico_glob_terno, terno_tuple_rit_calc, "terno", end_ts)
                            if isinstance(current_rit_val_glob_terno, (int, float)): min_rit_terno_glob = min(min_rit_terno_glob, current_rit_val_glob_terno); rit_valido_terno_glob_trovato = True
                    rit_da_memorizzare_terno = "N/A"
                    if rit_valido_terno_glob_trovato: rit_da_memorizzare_terno = int(min_rit_terno_glob)
                    else: rit_da_memorizzare_terno = "N/D"
                    ritardi_min_attuali_terni_individuali[terno_tuple_rit_calc] = rit_da_memorizzare_terno
                if num_top_terni_individuali_display > 0:
                    messaggi_out.append(f"  Top {num_top_terni_individuali_display} terni individuali (per copertura eventi spia):")
                    for i in range(num_top_terni_individuali_display):
                        terno_glob_tuple, count_glob_terno = migliori_terni_globali_raw[i]; terno_glob_str = format_ambo_terno(terno_glob_tuple)
                        perc_glob_terno = (count_glob_terno / n_trig_tot * 100) if n_trig_tot > 0 else 0.0
                        rit_individual_terno_val = ritardi_min_attuali_terni_individuali.get(terno_glob_tuple); rit_individual_display = ""
                        if rit_individual_terno_val is not None and rit_individual_terno_val not in ["N/A", "N/D"]: rit_individual_display = f" [Rit.Min.Att: {rit_individual_terno_val}]"
                        elif rit_individual_terno_val == "N/D": rit_individual_display = " [Rit.Min.Att: N/D]"
                        messaggi_out.append(f"    {i+1}. Terno {terno_glob_str}: Coperti {count_glob_terno} su {n_trig_tot} eventi spia ({perc_glob_terno:.1f}%){rit_individual_display}")
                        info_curr['migliori_terni_copertura_globale'].append({ "terno": terno_glob_str, "coperti": count_glob_terno, "totali": n_trig_tot, "percentuale": perc_glob_terno, "ritardo_min_attuale": rit_individual_terno_val })
                else: messaggi_out.append("    Nessun terno individuale con copertura significativa trovato.")
                info_curr['config_combinazioni_terni'] = {'num_top_terni_considerati': NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI}
                messaggi_out.append(f"\n\n=== MIGLIORI COMBINAZIONI DI TERNI PER COPERTURA GLOBALE (Ricerca tra Top {NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI} terni individuali) ===")
                max_copertura_raggiunta_terni_comb = -1
                migliore_k_per_terni_comb = None
                for k_val_terni in [3, 4, 5]:
                    info_key_comb_terni = f'migliore_combinazione_{k_val_terni}_terni'
                    risultato_combinazione_k_terni = trova_migliore_combinazione_copertura(terni_copertura_globale_eventi, k_val_terni, n_trig_tot,dizionario_ritardi_items_individuali=ritardi_min_attuali_terni_individuali,num_top_items_da_considerare=NUM_TOP_TERNI_DA_USARE_PER_COMBINAZIONI)
                    info_curr[info_key_comb_terni] = risultato_combinazione_k_terni
                    copertura_attuale_k_comb_terni = risultato_combinazione_k_terni.get("eventi_coperti", -1) if risultato_combinazione_k_terni else -1
                    if risultato_combinazione_k_terni and risultato_combinazione_k_terni.get("items_combinati_dettagli") and copertura_attuale_k_comb_terni >= 0:
                        messaggi_out.append(f"\n  --- Combinazione di {k_val_terni} Terni (per Log) ---")
                        for t_str_log_comb in risultato_combinazione_k_terni['items_combinati_dettagli']: messaggi_out.append(f"      - {t_str_log_comb}")
                        messaggi_out.append(f"    Copertura combinata: {copertura_attuale_k_comb_terni} su {n_trig_tot} eventi spia ({risultato_combinazione_k_terni['percentuale_copertura']:.1f}%)")
                        if copertura_attuale_k_comb_terni > max_copertura_raggiunta_terni_comb:
                            max_copertura_raggiunta_terni_comb = copertura_attuale_k_comb_terni
                            migliore_k_per_terni_comb = k_val_terni
                        elif copertura_attuale_k_comb_terni == max_copertura_raggiunta_terni_comb and migliore_k_per_terni_comb is None:
                            migliore_k_per_terni_comb = k_val_terni
                    elif risultato_combinazione_k_terni and risultato_combinazione_k_terni.get("messaggio"):
                         messaggi_out.append(f"\n  --- Combinazione di {k_val_terni} Terni ---"); messaggi_out.append(f"    {risultato_combinazione_k_terni['messaggio']}")
                    else:
                         messaggi_out.append(f"\n  --- Combinazione di {k_val_terni} Terni ---"); messaggi_out.append(f"    Nessuna combinazione significativa di {k_val_terni} terni trovata.")
                if migliore_k_per_terni_comb is not None:
                    copertura_ottimale_terni_comb = info_curr[f'migliore_combinazione_{migliore_k_per_terni_comb}_terni'].get('eventi_coperti', -1)
                    for k_successivo_terni in range(migliore_k_per_terni_comb + 1, 6):
                        info_key_successiva_terni = f'migliore_combinazione_{k_successivo_terni}_terni'
                        if info_curr.get(info_key_successiva_terni) and \
                           info_curr[info_key_successiva_terni].get('eventi_coperti', -1) <= copertura_ottimale_terni_comb:
                            info_curr[info_key_successiva_terni] = None
        # Fine sezioni globali per modalità successivi

        aggiorna_risultati_globali(ris_graf_loc,info_curr,modalita="successivi")
        if ris_graf_loc or (not all_date_trig and tipo_spia_scelto):
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali)
        # =============== FINE LOGICA MODALITA' "SUCCESSIVI" ===============

    elif modalita == "antecedenti":
        # =============== INIZIO LOGICA MODALITA' "ANTECEDENTI" (CON NUOVA COPERTURA E TERMINOLOGIA MODIFICATA) ===============
        ra_ant_idx = listbox_ruote_analisi_ant.curselection()
        if not ra_ant_idx:
            messagebox.showwarning("Manca Input", "Seleziona Ruota/e Analisi.")
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
            risultato_text.insert(tk.END, "Input Ruote Analisi Antecedenti mancante."); risultato_text.config(state=tk.NORMAL)
            return
        
        nomi_ra_ant = [listbox_ruote_analisi_ant.get(i) for i in ra_ant_idx]
        
        num_obj_raw = [e.get().strip() for e in entry_numeri_obiettivo if e.get().strip() and e.get().strip().isdigit() and 1 <= int(e.get().strip()) <= 90]
        num_obj = sorted(list(set(str(int(n)).zfill(2) for n in num_obj_raw)))
        if not num_obj:
            messagebox.showwarning("Manca Input", "Numeri Obiettivo non validi (1-90).")
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
            risultato_text.insert(tk.END, "Input Numeri Obiettivo mancante o non valido."); risultato_text.config(state=tk.NORMAL)
            return
        
        try:
            n_prec = int(estrazioni_entry_ant.get())
            assert n_prec >= 1
        except:
            messagebox.showerror("Input Invalido", f"N. Estrazioni Precedenti (1-{MAX_COLPI_GIOCO}) non valido.")
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
            risultato_text.insert(tk.END, "Input N. Estrazioni Precedenti non valido."); risultato_text.config(state=tk.NORMAL)
            return

        messaggi_out.append(f"--- Analisi Antecedenti (Marker) ---")
        messaggi_out.append(f"Numeri Obiettivo: {', '.join(num_obj)}")
        messaggi_out.append(f"Numero Estrazioni Precedenti Controllate: {n_prec}")
        messaggi_out.append(f"Periodo: {start_ts.strftime('%d/%m/%Y')} - {end_ts.strftime('%d/%m/%Y')}")
        messaggi_out.append("-" * 40)

        df_cache_ant = {}
        almeno_un_risultato_antecedente_significativo_globale = False 

        for nome_ra_ant_loop in nomi_ra_ant:
            df_ant_full = df_cache_ant.get(nome_ra_ant_loop)
            if df_ant_full is None:
                df_ant_full = carica_dati(file_ruote.get(nome_ra_ant_loop), start_ts, end_ts)
                df_cache_ant[nome_ra_ant_loop] = df_ant_full
            
            if df_ant_full is None or df_ant_full.empty:
                messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato storico nel periodo selezionato.")
                continue
            
            res_ant, err_ant = analizza_antecedenti(
                df_ruota=df_ant_full, 
                numeri_obiettivo=num_obj, 
                n_precedenti=n_prec, 
                nome_ruota=nome_ra_ant_loop
            )
            
            if err_ant:
                messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Errore: {err_ant}")
                continue

            msg_res_ant_corrente_per_ruota = "" 
            ruota_ha_risultati_significativi_locali = False

            if res_ant:
                msg_res_ant_corrente_per_ruota += f"\n=== Risultati Antecedenti per Ruota: {nome_ra_ant_loop.upper()} ==="
                msg_res_ant_corrente_per_ruota += f"\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Estrazioni Prec.: {res_ant['n_precedenti']})"
                # MODIFICATO TERMINOLOGIA
                msg_res_ant_corrente_per_ruota += f"\n(Occorrenze Obiettivo nel periodo: {res_ant['totale_occorrenze_obiettivo']} | Casi validi con antecedenti: {res_ant['base_presenza_antecedenti']})"

                if res_ant.get('presenza') and not res_ant['presenza']['top'].empty:
                    ruota_ha_risultati_significativi_locali = True
                    # MODIFICATO TERMINOLOGIA
                    msg_res_ant_corrente_per_ruota += f"\n\nTop Antecedenti per Presenza (su {res_ant['base_presenza_antecedenti']} casi validi):"
                    for i, (num_p, pres_p) in enumerate(res_ant['presenza']['top'].head(10).items()):
                        perc_pres_p_val = res_ant['presenza']['percentuali'].get(num_p, 0.0)
                        freq_p_val = res_ant['presenza']['frequenze'].get(num_p, 0)
                        msg_res_ant_corrente_per_ruota += f"\n  {i+1}. {num_p}: {pres_p} ({perc_pres_p_val:.1f}%) [Freq.Tot: {freq_p_val}]"
                else:
                    msg_res_ant_corrente_per_ruota += "\n\nNessun Top per Presenza individuale."

                if res_ant.get('frequenza') and not res_ant['frequenza']['top'].empty:
                    ruota_ha_risultati_significativi_locali = True
                    msg_res_ant_corrente_per_ruota += f"\n\nTop Antecedenti per Frequenza Totale:"
                    for i, (num_f, freq_f) in enumerate(res_ant['frequenza']['top'].head(10).items()):
                        perc_freq_f_val = res_ant['frequenza']['percentuali'].get(num_f, 0.0)
                        pres_f_val = res_ant['frequenza']['presenze'].get(num_f, 0)
                        # MODIFICATO TERMINOLOGIA
                        msg_res_ant_corrente_per_ruota += f"\n  {i+1}. {num_f}: {freq_f} ({perc_freq_f_val:.1f}%) [Pres. su Casi: {pres_f_val}]"
                else:
                    msg_res_ant_corrente_per_ruota += "\n\nNessun Top per Frequenza individuale."

                top_cop_ant_config = res_ant.get('top_copertura_combinata_antecedenti')
                if top_cop_ant_config:
                    ruota_ha_risultati_significativi_locali = True
                    # MODIFICATO TERMINOLOGIA
                    msg_res_ant_corrente_per_ruota += f"\n\n--- Configurazioni Antecedenti con Maggiore Copertura dell'Obiettivo ({', '.join(res_ant['numeri_obiettivo'])}) ---"
                    msg_res_ant_corrente_per_ruota += f"\n(Analisi su {res_ant['base_presenza_antecedenti']} casi validi dell'obiettivo con antecedenti)"
                    for item_cop_config in top_cop_ant_config:
                        k_val_c = item_cop_config['k']
                        numeri_str_list_c = item_cop_config['numeri']
                        casi_cop_c = item_cop_config['casi_coperti']
                        perc_cop_c = item_cop_config['percentuale_copertura']
                        numeri_display_str = ", ".join(numeri_str_list_c)
                        msg_res_ant_corrente_per_ruota += f"\n\n  Configurazione di {k_val_c} Antecedent{'e' if k_val_c == 1 else 'i'}: [{numeri_display_str}]"
                        # MODIFICATO TERMINOLOGIA
                        msg_res_ant_corrente_per_ruota += f"\n    Ha preceduto l'obiettivo (almeno un numero presente) in: {casi_cop_c} casi ({perc_cop_c:.1f}%)"
                elif res_ant['base_presenza_antecedenti'] > 0:
                     msg_res_ant_corrente_per_ruota += "\n\n--- Configurazioni Antecedenti con Maggiore Copertura: Non calcolate o nessun antecedente significativo per combinazioni."
                
                if ruota_ha_risultati_significativi_locali:
                    almeno_un_risultato_antecedente_significativo_globale = True
                    messaggi_out.append(msg_res_ant_corrente_per_ruota)
                else: 
                    messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato antecedente significativo trovato (Presenza, Frequenza o Copertura).")
            else:
                messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Analisi antecedenti non ha prodotto risultati validi.")
            messaggi_out.append("\n" + ("- " * 20))

        aggiorna_risultati_globali([], {}, modalita="antecedenti")
        # =============== FINE LOGICA MODALITA' "ANTECEDENTI" ===============

    final_output = "\n".join(messaggi_out) if messaggi_out else "Nessun risultato o elaborazione completata senza output specifico."
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, final_output)
    risultato_text.see("1.0")

    if modalita == "antecedenti" and almeno_un_risultato_antecedente_significativo_globale:
        mostra_popup_testo_semplice("Riepilogo Analisi Numeri Antecedenti", final_output)
    elif modalita == "antecedenti" and not almeno_un_risultato_antecedente_significativo_globale and messaggi_out:
        mostra_popup_testo_semplice("Info Analisi Numeri Antecedenti", final_output)
    
# =============================================================================
def verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_ruote_verifica, n_verifiche, start_ts, end_ts, titolo_sezione="VERIFICA MISTA SU TRIGGER"): # Nome parametro trigger qui non cambia
    if not date_triggers or not combinazioni_utente or not nomi_ruote_verifica:
        return "Errore: Dati input mancanti per verifica utente su triggers."
    
    estratti_u, ambi_u_tpl, terni_u_tpl, quaterne_u_tpl, cinquine_u_tpl = [], [], [], [], []
    if isinstance(combinazioni_utente, dict):
        estratti_u = sorted(list(set(combinazioni_utente.get('estratto', []))))
        ambi_u_tpl = sorted(list(set(tuple(sorted(a)) for a in combinazioni_utente.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
        terni_u_tpl = sorted(list(set(tuple(sorted(t)) for t in combinazioni_utente.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
        quaterne_u_tpl = sorted(list(set(tuple(sorted(q)) for q in combinazioni_utente.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
        cinquine_u_tpl = sorted(list(set(tuple(sorted(c)) for c in combinazioni_utente.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))

    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver = {}; ruote_valide = []
    for nome_rv_loop in nomi_ruote_verifica:
        # Carica i dati completi per la ruota, filtrando DOPO per le date rilevanti
        # Questo è importante per avere estrazioni anche dopo end_ts se n_verifiche va oltre
        df_ver = carica_dati(file_ruote.get(nome_rv_loop), start_date=None, end_date=None) # Carica tutto
        if df_ver is not None and not df_ver.empty:
            df_cache_ver[nome_rv_loop] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
            ruote_valide.append(nome_rv_loop)

    if not ruote_valide: return "Errore: Nessuna ruota di verifica valida per caricare i dati."

    sorted_date_triggers = sorted(list(date_triggers))
    num_casi_base_originali = len(sorted_date_triggers) # Numero di "eventi spia" originali
    
    if num_casi_base_originali == 0: return "Nessun caso trigger/spia da verificare."

    # Il titolo della sezione viene passato come argomento
    out = [f"\n\n{titolo_sezione}"] 
    # MODIFICATO: Etichetta per il numero di casi
    out.append(f"Numero di casi di base: {num_casi_base_originali}") 
    out.append(f"Ruote di verifica considerate: {', '.join(ruote_valide) or 'Nessuna'}")

    esiti_dettagliati_per_item = {}; 
    items_config_da_verificare = []
    if estratti_u: items_config_da_verificare.extend([('estratto', e) for e in estratti_u])
    if ambi_u_tpl: items_config_da_verificare.extend([('ambo', a) for a in ambi_u_tpl])
    if terni_u_tpl: items_config_da_verificare.extend([('terno', t) for t in terni_u_tpl])
    if quaterne_u_tpl: items_config_da_verificare.extend([('quaterna', q) for q in quaterne_u_tpl])
    if cinquine_u_tpl: items_config_da_verificare.extend([('cinquina', c) for c in cinquine_u_tpl])

    if not items_config_da_verificare:
        out.append("\nNessuna combinazione utente valida da verificare (estratti, ambi, terni, ecc.).")
        return "\n".join(out)

    sfaldamenti_totali_per_item_ruota = {}

    for tipo_sorte_item, item_val_originale in items_config_da_verificare:
        # Assicura che gli item siano nel formato corretto (stringa per estratto, tupla per combinazioni)
        item_val_per_check = item_val_originale # Per estratto è già stringa
        if isinstance(item_val_originale, tuple): # Per ambo, terno etc.
            item_val_per_check = tuple(sorted(str(n).zfill(2) for n in item_val_originale))
        
        item_str_key = format_ambo_terno(item_val_per_check) # Chiave stringa per i dizionari
        
        esiti_dettagliati_per_item[item_str_key] = []
        num_eventi_coperti_per_item = 0 # Conteggio dei "casi di base" coperti
        total_actual_hits_for_item = 0 
        sfaldamenti_totali_per_item_ruota[item_str_key] = Counter()

        for data_t_trigger in sorted_date_triggers: # Itera sui casi di base
            dettagli_uscita_per_questo_caso_lista = []
            caso_coperto_in_questo_ciclo = False
            max_colpi_effettivi_per_questo_caso = 0 # Per tracciare l'esito "IN CORSO"

            for nome_rv in ruote_valide:
                df_v = df_cache_ver.get(nome_rv)
                if df_v is None: continue # Non dovrebbe succedere se ruote_valide è corretto
                
                date_s_v = df_v['Data']
                try:
                    # Cerca l'indice della prima estrazione DOPO la data del trigger
                    start_idx = date_s_v.searchsorted(data_t_trigger, side='right') 
                except Exception: # Dovrebbe essere raro con dati ordinati
                    continue 
                if start_idx >= len(date_s_v): # Nessuna estrazione dopo il trigger su questa ruota
                    max_colpi_effettivi_per_questo_caso = max(max_colpi_effettivi_per_questo_caso, 0)
                    continue

                df_finestra_verifica = df_v.iloc[start_idx : start_idx + n_verifiche]
                max_colpi_effettivi_per_questo_caso = max(max_colpi_effettivi_per_questo_caso, len(df_finestra_verifica))

                if not df_finestra_verifica.empty:
                    for colpo_idx, (_, row_estrazione) in enumerate(df_finestra_verifica.iterrows(), 1):
                        data_uscita_corrente = row_estrazione['Data'].date()
                        numeri_estratti_nella_riga = [str(row_estrazione[col]).zfill(2) for col in cols_num if pd.notna(row_estrazione[col])]
                        set_numeri_estratti_riga = set(numeri_estratti_nella_riga)
                        
                        match_trovato = False
                        if tipo_sorte_item == 'estratto': # item_val_per_check è una stringa
                            if item_val_per_check in set_numeri_estratti_riga: match_trovato = True
                        elif isinstance(item_val_per_check, tuple): # Per ambo, terno, ecc. item_val_per_check è una tupla di stringhe
                            if set(item_val_per_check).issubset(set_numeri_estratti_riga): match_trovato = True
                        
                        if match_trovato:
                            total_actual_hits_for_item += 1
                            dettagli_uscita_per_questo_caso_lista.append(f"{nome_rv} @ C{colpo_idx} ({data_uscita_corrente.strftime('%d/%m')})")
                            caso_coperto_in_questo_ciclo = True
                            
                            # Conta lo sfaldamento su ruota solo la prima volta che l'item esce per questo caso trigger
                            if nome_rv not in sfaldamenti_totali_per_item_ruota[item_str_key]:
                               sfaldamenti_totali_per_item_ruota[item_str_key][nome_rv] +=1
                            # Non aggiungere più volte lo stesso nome_rv a sfaldamenti_totali_per_item_ruota[item_str_key]
                            # Il contatore fa già questo implicitamente se lo incrementi solo
                            break # Trovato per questa ruota e caso, passa al prossimo caso trigger
                if caso_coperto_in_questo_ciclo: # Se è stato trovato su una ruota, non cercare su altre per questo caso trigger
                    break
            
            esito_stringa_per_questo_caso = ""
            if caso_coperto_in_questo_ciclo:
                esito_stringa_per_questo_caso = "[" + "; ".join(dettagli_uscita_per_questo_caso_lista) + "]"
                num_eventi_coperti_per_item +=1
            elif max_colpi_effettivi_per_questo_caso < n_verifiche and max_colpi_effettivi_per_questo_caso >= 0 : # Se ci sono state estrazioni ma meno di n_verifiche
                esito_stringa_per_questo_caso = f"IN CORSO (max {max_colpi_effettivi_per_questo_caso}/{n_verifiche} colpi analizzabili)"
            else: # max_colpi_effettivi_per_questo_caso == n_verifiche (o == 0 se nessuna estrazione trovata dopo il trigger)
                esito_stringa_per_questo_caso = "NON USCITO"
            
            esiti_dettagliati_per_item[item_str_key].append(f"    {data_t_trigger.strftime('%d/%m/%y')}: {esito_stringa_per_questo_caso}")

        out.append(f"\n--- Esiti {tipo_sorte_item.upper()} ---");
        out.append(f"  - {item_str_key}:");
        out.extend(esiti_dettagliati_per_item[item_str_key])
        
        out.append(f"\n    RIEPILOGO SFALDAMENTI SU RUOTA per {item_str_key.upper()}:")
        if not ruote_valide:
             out.append("      Nessuna ruota di verifica attiva.")
        else:
            almeno_uno_sfaldamento_su_ruota = False
            for nome_rv_riepilogo in ruote_valide:
                conteggio_sf = sfaldamenti_totali_per_item_ruota[item_str_key].get(nome_rv_riepilogo, 0)
                if conteggio_sf > 0: almeno_uno_sfaldamento_su_ruota = True
                out.append(f"      - {nome_rv_riepilogo}: {conteggio_sf} volt{'a' if conteggio_sf == 1 else 'e'}")
            if not almeno_uno_sfaldamento_su_ruota and ruote_valide:
                 out.append("      Nessuno sfaldamento registrato su alcuna ruota per questo item.")

        if num_casi_base_originali == 0:
            out.append(f"    RIEPILOGO {item_str_key}: Nessun caso base registrato per l'analisi.")
        else:
            perc_copertura_casi_base = (num_eventi_coperti_per_item / num_casi_base_originali * 100) if num_casi_base_originali > 0 else 0.0
            out.append(f"    RIEPILOGO {item_str_key}: Ha coperto {num_eventi_coperti_per_item} su {num_casi_base_originali} casi di base ({perc_copertura_casi_base:.1f}%). Uscite totali nelle finestre: {total_actual_hits_for_item}.")
            
    return "\n".join(out)

def verifica_esiti_futuri(top_combinati_input, nomi_ruote_verifica, data_fine_analisi, n_colpi_futuri):
    if not top_combinati_input or not any(top_combinati_input.values()) or not nomi_ruote_verifica or data_fine_analisi is None or n_colpi_futuri <= 0:
        return "Errore: Input invalidi per verifica_esiti_futuri (post-analisi)."
    estratti_items = sorted(list(set(top_combinati_input.get('estratto', []))))
    ambi_items = sorted(list(set(tuple(sorted(a)) for a in top_combinati_input.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_items = sorted(list(set(tuple(sorted(t)) for t in top_combinati_input.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
    quaterne_items = sorted(list(set(tuple(sorted(q)) for q in top_combinati_input.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_items = sorted(list(set(tuple(sorted(c)) for c in top_combinati_input.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))
    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver_fut = {}; ruote_con_dati_fut = []
    for nome_rv in nomi_ruote_verifica:
        df_ver_full = carica_dati(file_ruote.get(nome_rv), start_date=None, end_date=None)
        if df_ver_full is None or df_ver_full.empty: continue
        df_ver_fut_ruota = df_ver_full[df_ver_full['Data'] > data_fine_analisi].copy().sort_values(by='Data').reset_index(drop=True)
        df_fin_fut_ruota = df_ver_fut_ruota.head(n_colpi_futuri)
        if not df_fin_fut_ruota.empty: df_cache_ver_fut[nome_rv] = df_fin_fut_ruota; ruote_con_dati_fut.append(nome_rv)
    if not ruote_con_dati_fut: return f"Nessuna estrazione trovata su nessuna ruota di verifica dopo {data_fine_analisi.strftime('%d/%m/%Y')} per {n_colpi_futuri} colpi."
    num_colpi_globalmente_analizzabili = n_colpi_futuri
    if ruote_con_dati_fut: min_len = float('inf');_=[(min_len:=min(min_len,len(df_cache_ver_fut.get(r,pd.DataFrame())))) for r in ruote_con_dati_fut];num_colpi_globalmente_analizzabili=min_len
    else: num_colpi_globalmente_analizzabili = 0
    hits_registrati = {'estratto':{e:None for e in estratti_items},'ambo':{a:None for a in ambi_items},'terno':{t:None for t in terni_items},'quaterna':{q:None for q in quaterne_items},'cinquina':{c:None for c in cinquine_items}}
    for nome_rv in ruote_con_dati_fut:
        df_finestra_ruota = df_cache_ver_fut[nome_rv]
        for colpo_idx, (_, row) in enumerate(df_finestra_ruota.iterrows(), 1):
            data_estrazione_corrente = row['Data'].date(); numeri_estratti_riga = [row[col] for col in cols_num if pd.notna(row[col])]; numeri_sortati_riga = sorted(numeri_estratti_riga)
            if not numeri_sortati_riga: continue
            set_numeri_riga_formattati = set(numeri_sortati_riga)
            if estratti_items:
                for item_e_orig in estratti_items:
                    if hits_registrati['estratto'][item_e_orig] is None and format_ambo_terno(item_e_orig) in set_numeri_riga_formattati: hits_registrati['estratto'][item_e_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if ambi_items and len(numeri_sortati_riga) >= 2:
                for item_a_orig in ambi_items:
                    if hits_registrati['ambo'][item_a_orig] is None and set(item_a_orig).issubset(set_numeri_riga_formattati): hits_registrati['ambo'][item_a_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if terni_items and len(numeri_sortati_riga) >= 3:
                for item_t_orig in terni_items:
                    if hits_registrati['terno'][item_t_orig] is None and set(item_t_orig).issubset(set_numeri_riga_formattati): hits_registrati['terno'][item_t_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if quaterne_items and len(numeri_sortati_riga) >= 4:
                for item_q_orig in quaterne_items:
                     if hits_registrati['quaterna'][item_q_orig] is None and set(item_q_orig).issubset(set_numeri_riga_formattati): hits_registrati['quaterna'][item_q_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if cinquine_items and len(numeri_sortati_riga) >= 5:
                for item_c_orig in cinquine_items:
                    if hits_registrati['cinquina'][item_c_orig] is None and set(item_c_orig).issubset(set_numeri_riga_formattati): hits_registrati['cinquina'][item_c_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
    out = [f"\n\n=== VERIFICA ESITI FUTURI (POST-ANALISI) ({n_colpi_futuri} Colpi dopo {data_fine_analisi.strftime('%d/%m/%Y')}) ==="]
    out.append(f"Ruote verificate con dati futuri disponibili: {', '.join(ruote_con_dati_fut) or 'Nessuna'}")
    if ruote_con_dati_fut: out.append(f"(Analisi basata su un minimo di {num_colpi_globalmente_analizzabili} colpi disponibili globalmente su queste ruote)")
    sorti_config = [('estratto', estratti_items), ('ambo', ambi_items), ('terno', terni_items),('quaterna', quaterne_items), ('cinquina', cinquine_items)]
    for tipo_sorte, lista_items_sorte in sorti_config:
        if not lista_items_sorte: continue
        out.append(f"\n--- Esiti Futuri {tipo_sorte.upper()} ---"); almeno_un_hit_per_sorte = False
        for item_da_verificare_orig in lista_items_sorte:
            item_str_formattato = format_ambo_terno(item_da_verificare_orig); dettaglio_hit = hits_registrati[tipo_sorte].get(item_da_verificare_orig)
            if dettaglio_hit: almeno_un_hit_per_sorte = True; d_ruota, d_colpo, d_data = dettaglio_hit; out.append(f"    - {item_str_formattato}: USCITO -> {d_ruota} @ C{d_colpo} ({d_data.strftime('%d/%m/%Y')})")
            else:
                if num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut : out.append(f"    - {item_str_formattato}: IN CORSO (analizzati min {num_colpi_globalmente_analizzabili}/{n_colpi_futuri} colpi)")
                else: out.append(f"    - {item_str_formattato}: NON uscito")
        if not almeno_un_hit_per_sorte and lista_items_sorte and not (num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut): out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito nei colpi futuri analizzati.")
    return "\n".join(out)


# MODIFICATO per gestione stato risultato_text per copia/incolla
def esegui_verifica_futura():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica_futura, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, "\n\nVerifica esiti futuri (post-analisi) per MIGLIORI COPERTURE in corso..."); # Titolo modificato
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    nomi_rv = info_ricerca_globale.get('ruote_verifica')
    data_fine = info_ricerca_globale.get('end_date')

    if not all([nomi_rv, data_fine]):
        messagebox.showerror("Errore Verifica Futura", "Dati analisi 'Successivi' (Ruote verifica, Data Fine) mancanti.");
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Futura.");risultato_text.config(state=tk.NORMAL); return

    # --- NUOVA LOGICA PER ESTRARRE I NUMERI DALLE MIGLIORI COPERTURE ---
    items_da_verificare = {'estratto': [], 'ambo': [], 'terno': []}

    # Estratti
    estratti_100 = info_ricerca_globale.get('combinazione_ottimale_estratti_100')
    estratti_parz = info_ricerca_globale.get('migliore_combinazione_parziale_estratti')
    if estratti_100 and estratti_100.get('estratti'):
        items_da_verificare['estratto'].extend(estratti_100['estratti'])
    elif estratti_parz and estratti_parz.get('estratti'):
        items_da_verificare['estratto'].extend(estratti_parz['estratti'])
    
    # Ambi
    ambi_100 = info_ricerca_globale.get('combinazione_ottimale_copertura_100') # Questo è per gli ambi
    ambi_parz = info_ricerca_globale.get('migliore_combinazione_parziale')     # Questo è per gli ambi
    if ambi_100 and ambi_100.get('ambi'):
        # 'ambi' qui contiene stringhe tipo "num1-num2", dobbiamo splittarle
        items_da_verificare['ambo'].extend([tuple(sorted(a.split('-'))) for a in ambi_100['ambi']])
    elif ambi_parz and ambi_parz.get('ambi'):
        items_da_verificare['ambo'].extend([tuple(sorted(a.split('-'))) for a in ambi_parz['ambi']])

    # Terni (trova il k che ha dato la migliore copertura)
    miglior_k_terni = None
    max_cop_terni = -1
    risultato_miglior_k_terni = None
    for k_check in [3, 4, 5]:
        info_key = f'migliore_combinazione_{k_check}_terni'
        comb_check = info_ricerca_globale.get(info_key)
        if comb_check and comb_check.get('eventi_coperti', -1) > max_cop_terni:
            max_cop_terni = comb_check['eventi_coperti']
            miglior_k_terni = k_check
            risultato_miglior_k_terni = comb_check
        elif comb_check and comb_check.get('eventi_coperti', -1) == max_cop_terni and (miglior_k_terni is None or k_check < miglior_k_terni) :
            miglior_k_terni = k_check # Preferisci il k più piccolo a parità di copertura
            risultato_miglior_k_terni = comb_check


    if risultato_miglior_k_terni and risultato_miglior_k_terni.get('items_combinati_str'):
        # 'items_combinati_str' contiene stringhe tipo "n1-n2-n3"
        items_da_verificare['terno'].extend([tuple(sorted(t.split('-'))) for t in risultato_miglior_k_terni['items_combinati_str']])
    
    # Rimuovi duplicati e assicurati che siano tuple per ambi/terni e stringhe per estratti
    items_da_verificare['estratto'] = sorted(list(set(items_da_verificare['estratto'])))
    items_da_verificare['ambo'] = sorted(list(set(items_da_verificare['ambo'])))
    items_da_verificare['terno'] = sorted(list(set(items_da_verificare['terno'])))

    if not any(items_da_verificare.values()): # Se nessuna lista ha elementi
        messagebox.showinfo("Verifica Futura", "Nessun elemento trovato dalle 'Migliori Coperture' per la verifica.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessun elemento da Migliori Coperture.");risultato_text.config(state=tk.NORMAL); return
    
    # --- FINE NUOVA LOGICA ---

    try: n_colpi_fut = int(estrazioni_entry_verifica_futura.get()); assert 1 <= n_colpi_fut <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Futura (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica = ""
    try:
        # La funzione verifica_esiti_futuri si aspetta un dizionario come 'top_combinati_input'
        # quindi passiamo il nostro 'items_da_verificare'
        res_str = verifica_esiti_futuri(items_da_verificare, nomi_rv, data_fine, n_colpi_fut)
        output_verifica = res_str
        if res_str and "Errore" not in res_str and "Nessuna estrazione trovata" not in res_str :
            mostra_popup_testo_semplice("Riepilogo Verifica Predittiva (Migliori Coperture)", res_str) # Titolo popup modificato
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
    risultato_text.insert(tk.END, "\n\nVerifica mista (combinazioni utente su trigger spia) in corso...");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    input_text = text_combinazioni_miste.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Input Invalido", "Nessuna combinazione inserita.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione inserita.");risultato_text.config(state=tk.NORMAL); return

    combinazioni_sets = {'estratto': set(),'ambo': set(),'terno': set(),'quaterna': set(),'cinquina': set()}; righe_input_originali = []
    for riga_idx, riga in enumerate(input_text.splitlines()):
        riga_proc = riga.strip();
        if not riga_proc: continue
        righe_input_originali.append(riga_proc) # Memorizza la riga originale per l'output
        try:
            numeri_str = riga_proc.split('-') if '-' in riga_proc else riga_proc.split(); # Gestisce sia '-' che spazio
            numeri_int = [int(n.strip()) for n in numeri_str if n.strip().isdigit()]
            if not numeri_int: raise ValueError("Nessun numero valido sulla riga.")
            if not all(1<=n<=90 for n in numeri_int): raise ValueError("Numeri fuori range (1-90).")
            if len(set(numeri_int))!=len(numeri_int): raise ValueError("Numeri duplicati nella stessa combinazione.") # Chiarito il messaggio
            if not (1<=len(numeri_int)<=5): raise ValueError("Inserire da 1 a 5 numeri per combinazione.")
            
            numeri_validi_zfill = sorted([str(n).zfill(2) for n in numeri_int]); num_elementi = len(numeri_validi_zfill)
            
            if num_elementi == 1: combinazioni_sets['estratto'].add(numeri_validi_zfill[0])
            elif num_elementi == 2: combinazioni_sets['ambo'].add(tuple(numeri_validi_zfill))
            elif num_elementi == 3: combinazioni_sets['terno'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)] # Sviluppa ambi
            elif num_elementi == 4: combinazioni_sets['quaterna'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['terno'].add(tuple(sorted(tc))) for tc in itertools.combinations(numeri_validi_zfill,3)]; _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)] # Sviluppa terni e ambi
            elif num_elementi == 5: combinazioni_sets['cinquina'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['quaterna'].add(tuple(sorted(qc))) for qc in itertools.combinations(numeri_validi_zfill,4)]; _=[combinazioni_sets['terno'].add(tuple(sorted(tc))) for tc in itertools.combinations(numeri_validi_zfill,3)]; _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)] # Sviluppa quaterne, terni, ambi
        except ValueError as ve:
            messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {ve}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return
        except Exception as e_parse: # Catch generico per altri errori di parsing
            messagebox.showerror("Input Invalido",f"Errore generico durante l'elaborazione della riga '{riga_proc}': {e_parse}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return

    combinazioni_utente = {k: sorted(list(v)) for k, v in combinazioni_sets.items() if v} # Filtra sorti vuote
    
    if not any(combinazioni_utente.values()): # Se nessuna combinazione valida è stata estratta
        messagebox.showerror("Input Invalido","Nessuna combinazione valida estratta dall'input.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione valida estratta.");risultato_text.config(state=tk.NORMAL);return

    date_triggers = info_ricerca_globale.get('date_trigger_ordinate'); nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date'); end_ts = info_ricerca_globale.get('end_date')
    
    # Recupera la spia originale per il titolo
    numeri_spia_originali_vm = info_ricerca_globale.get('numeri_spia_input', []) # Uso un nome diverso per evitare conflitti
    spia_display_originale_vm = ""
    tipo_spia_usato_vm = info_ricerca_globale.get('tipo_spia_usato', 'N/D').upper()
    if tipo_spia_usato_vm == "ESTRATTO_POSIZIONALE":
        num_spia_list_vm = numeri_spia_originali_vm if isinstance(numeri_spia_originali_vm, list) else [numeri_spia_originali_vm]
        num_spia_vm = num_spia_list_vm[0] if num_spia_list_vm else "N/D"
        pos_spia_vm = info_ricerca_globale.get('posizione_spia_input', "N/D")
        spia_display_originale_vm = f"{num_spia_vm} in {pos_spia_vm}a pos."
    elif isinstance(numeri_spia_originali_vm, tuple): spia_display_originale_vm = "-".join(map(str, numeri_spia_originali_vm))
    elif isinstance(numeri_spia_originali_vm, list): spia_display_originale_vm = ", ".join(map(str, numeri_spia_originali_vm))
    else: spia_display_originale_vm = str(numeri_spia_originali_vm)


    if not all([date_triggers, nomi_rv, start_ts is not None, end_ts is not None]):
        messagebox.showerror("Errore Verifica Mista", "Dati analisi 'Successivi' mancanti (date trigger, ruote verifica, periodo).");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Mista (dati analisi precedente mancanti).");risultato_text.config(state=tk.NORMAL); return
    try:
        n_colpi_misti = int(estrazioni_entry_verifica_mista.get()); assert 1 <= n_colpi_misti <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Mista (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi Verifica Mista non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica_mista = ""
    try:
        # MODIFICATO: Titolo della sezione nel report
        titolo_output_report = f"VERIFICA MISTA (COMBINAZIONI UTENTE) - Dopo Spia: {spia_display_originale_vm} ({n_colpi_misti} Colpi dopo ogni caso)"
        
        res_str = verifica_esiti_utente_su_triggers(
            date_triggers, 
            combinazioni_utente, 
            nomi_rv, 
            n_colpi_misti, 
            start_ts, 
            end_ts, 
            titolo_sezione=titolo_output_report # Passa il titolo aggiornato
        )
        
        # MODIFICATO: Etichetta per l'input utente
        summary_input = "\nNumero/i scelti dall'utente (righe elaborate):\n" + "\n".join([f"  - {r}" for r in righe_input_originali])
        
        # Inserisci il summary dell'input utente all'inizio del report generato da verifica_esiti_utente_su_triggers
        lines_res = res_str.splitlines()
        
        # Trova la riga del titolo della sezione per inserire il summary_input dopo
        indice_inserimento_summary = 0
        for i, linea in enumerate(lines_res):
            if titolo_output_report in linea: # Cerca la riga del titolo
                indice_inserimento_summary = i + 1 # Inserisci dopo la riga del titolo
                break
        
        final_output_lines = lines_res[:indice_inserimento_summary] + [summary_input] + lines_res[indice_inserimento_summary:]
        output_verifica_mista = "\n".join(final_output_lines)

        if output_verifica_mista and "Errore" not in output_verifica_mista and "Nessun caso" not in output_verifica_mista: # "Nessun caso" per "Nessun caso trigger"
            # Titolo del popup
            titolo_popup_vm = f"Riepilogo Verifica Mista (Spia: {spia_display_originale_vm})"
            mostra_popup_testo_semplice(titolo_popup_vm, output_verifica_mista)
    except Exception as e:
        output_verifica_mista = f"\nErrore durante la verifica mista: {e}"
        traceback.print_exc()

    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, output_verifica_mista)
    risultato_text.see(tk.END) # Scrolla alla fine


# =============================================================================
# Funzione Wrapper per Visualizza Grafici (INVARIATA)
# =============================================================================
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and bool(risultati_globali) and any(r[2] for r in risultati_globali if len(r)>2):
        valid_res = [r for r in risultati_globali if r[2] is not None]
        if valid_res: visualizza_grafici(valid_res, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni',5))
        else: messagebox.showinfo("Grafici", "Nessun risultato valido per grafici.")
    else: messagebox.showinfo("Grafici", "Esegui 'Cerca Successivi' con risultati validi prima.")


# =============================================================================
# FUNZIONI PER NUMERI SIMPATICI (Logica INVARIATA, gestione risultato_text MODIFICATA)
# =============================================================================
def trova_abbinamenti_numero_target(numero_target_str, nomi_ruote_ricerca, start_ts, end_ts, top_n_simpatici, max_k_ambi_copertura=5): # Nuovo parametro per copertura ambi
    global file_ruote
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']

    if not numero_target_str or not (numero_target_str.isdigit() and 1 <= int(numero_target_str) <= 90):
        return None, "Numero target non valido (deve essere 1-90).", 0, [], {} # Aggiunti valori di ritorno vuoti

    numero_target_zfill = numero_target_str.zfill(2)
    abbinamenti_counter = Counter()
    occorrenze_target = 0
    ruote_effettivamente_analizzate = []
    
    # NUOVO: Per l'analisi di copertura degli ambi
    date_uscite_target_con_ambi_effettivi = {} # Dizionario: {data_uscita_target: set_di_ambi_usciti_con_target_quel_giorno}

    for nome_ruota in nomi_ruote_ricerca:
        if nome_ruota not in file_ruote:
            print(f"Attenzione: File per ruota {nome_ruota} non trovato. Saltata.")
            continue
        df_ruota = carica_dati(file_ruote.get(nome_ruota), start_date=start_ts, end_date=end_ts)
        if df_ruota is None or df_ruota.empty:
            continue
        
        ruote_effettivamente_analizzate.append(nome_ruota)
        
        for data_estrazione, row in df_ruota.set_index('Data').iterrows(): # Iteriamo sulla data per facilità
            numeri_estratti_riga_originali = [row[col] for col in colonne_numeri if pd.notna(row[col])]
            # Assicuriamoci che siano stringhe zfillate
            numeri_estratti_riga_zfill = sorted([str(n).zfill(2) for n in numeri_estratti_riga_originali])

            if numero_target_zfill in numeri_estratti_riga_zfill:
                occorrenze_target += 1
                altri_numeri_nella_stessa_estrazione = [n for n in numeri_estratti_riga_zfill if n != numero_target_zfill]
                
                if altri_numeri_nella_stessa_estrazione:
                    abbinamenti_counter.update(altri_numeri_nella_stessa_estrazione)

                # NUOVO: Raccogli gli ambi usciti con il target in questa estrazione
                ambi_usciti_con_target_in_questa_estrazione = set()
                for altro_numero in altri_numeri_nella_stessa_estrazione:
                    ambo = tuple(sorted((numero_target_zfill, altro_numero)))
                    ambi_usciti_con_target_in_questa_estrazione.add(ambo)
                
                if ambi_usciti_con_target_in_questa_estrazione:
                    # Usiamo pd.Timestamp per coerenza con le date dei trigger spia
                    data_ts = pd.Timestamp(data_estrazione) 
                    if data_ts not in date_uscite_target_con_ambi_effettivi:
                        date_uscite_target_con_ambi_effettivi[data_ts] = set()
                    date_uscite_target_con_ambi_effettivi[data_ts].update(ambi_usciti_con_target_in_questa_estrazione)

    if not ruote_effettivamente_analizzate:
        return None, "Nessuna ruota selezionata conteneva dati validi.", 0, [], {}

    if occorrenze_target == 0:
        return [], f"Numero target '{numero_target_zfill}' non trovato.", 0, [], {}

    top_simpatici_list = abbinamenti_counter.most_common(top_n_simpatici)
    
    # --- NUOVO: Calcolo Copertura Combinata Ambi Simpatici ---
    risultati_copertura_ambi_simpatici = []
    
    # Ambi candidati: formati da numero_target + ciascuno dei top_n_simpatici
    ambi_candidati_da_simpatici = []
    if top_simpatici_list:
        for num_simp, _ in top_simpatici_list: # num_simp è già stringa zfillata
            ambo_candidato = tuple(sorted((numero_target_zfill, num_simp)))
            ambi_candidati_da_simpatici.append(ambo_candidato)
    
    # Rimuovi duplicati se un simpatico genera un ambo già presente (improbabile ma per sicurezza)
    ambi_candidati_da_simpatici = sorted(list(set(ambi_candidati_da_simpatici)))

    date_trigger_per_copertura_ambi = list(date_uscite_target_con_ambi_effettivi.keys())
    num_casi_target_con_ambi = len(date_trigger_per_copertura_ambi)

    if ambi_candidati_da_simpatici and num_casi_target_con_ambi > 0:
        copertura_100_raggiunta_ambi = False
        for k_ambi in range(1, min(max_k_ambi_copertura, len(ambi_candidati_da_simpatici)) + 1):
            if len(ambi_candidati_da_simpatici) < k_ambi:
                break
            
            migliore_combinazione_k_ambi_tuple = None
            max_casi_coperti_k_ambi = -1

            for combo_ambi_tuple_di_tuple in itertools.combinations(ambi_candidati_da_simpatici, k_ambi):
                # combo_ambi_tuple_di_tuple è una tupla di tuple ambo, es. (( (08,36), (08,67) ))
                
                casi_coperti_da_questa_combo_ambi = 0
                for data_uscita_target in date_trigger_per_copertura_ambi:
                    ambi_effettivi_quel_giorno = date_uscite_target_con_ambi_effettivi.get(data_uscita_target, set())
                    # La combinazione copre se ALMENO UNO degli ambi della combinazione è tra quelli effettivi
                    coperto_questo_giorno = False
                    for ambo_nella_combo in combo_ambi_tuple_di_tuple:
                        if ambo_nella_combo in ambi_effettivi_quel_giorno:
                            coperto_questo_giorno = True
                            break
                    if coperto_questo_giorno:
                        casi_coperti_da_questa_combo_ambi += 1
                
                if casi_coperti_da_questa_combo_ambi > max_casi_coperti_k_ambi:
                    max_casi_coperti_k_ambi = casi_coperti_da_questa_combo_ambi
                    migliore_combinazione_k_ambi_tuple = combo_ambi_tuple_di_tuple # Tupla di tuple ambo
                elif casi_coperti_da_questa_combo_ambi == max_casi_coperti_k_ambi and migliore_combinazione_k_ambi_tuple:
                    # Opzionale: preferenza lessicografica
                    if combo_ambi_tuple_di_tuple < migliore_combinazione_k_ambi_tuple:
                        migliore_combinazione_k_ambi_tuple = combo_ambi_tuple_di_tuple
            
            if migliore_combinazione_k_ambi_tuple is not None and max_casi_coperti_k_ambi >= 0:
                perc_copertura_k_ambi = (max_casi_coperti_k_ambi / num_casi_target_con_ambi * 100) if num_casi_target_con_ambi > 0 else 0.0
                # Formatta gli ambi per la visualizzazione
                ambi_str_list = sorted([format_ambo_terno(ambo_t) for ambo_t in migliore_combinazione_k_ambi_tuple])
                
                risultati_copertura_ambi_simpatici.append({
                    "k_ambi": k_ambi,
                    "ambi_combinati": ambi_str_list, # Lista di stringhe ambo "XX-YY"
                    "casi_target_coperti": max_casi_coperti_k_ambi,
                    "percentuale_copertura": round(perc_copertura_k_ambi, 1)
                })
                if round(perc_copertura_k_ambi, 1) >= 100.0:
                    copertura_100_raggiunta_ambi = True
                    break # Esci dal loop 'for k_ambi...'
            
            if copertura_100_raggiunta_ambi:
                break
    # --- FINE NUOVA LOGICA COPERTURA AMBI ---

    # Restituisci i numeri simpatici, il messaggio, le occorrenze E i risultati della copertura ambi
    return top_simpatici_list, None, occorrenze_target, risultati_copertura_ambi_simpatici, date_uscite_target_con_ambi_effettivi

def esegui_ricerca_numeri_simpatici():
    global risultato_text, root, entry_numero_target_simpatici, listbox_ruote_simpatici
    global entry_top_n_simpatici, start_date_entry, end_date_entry

    if not mappa_file_ruote() or not file_ruote:
        messagebox.showerror("Errore Cartella", "Impossibile leggere i file dalla cartella.\nAssicurati di aver selezionato una cartella con 'Sfoglia...'.")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
        risultato_text.insert(tk.END, "Errore: Cartella o file non validi."); risultato_text.config(state=tk.NORMAL)
        return

    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "Ricerca Numeri Simpatici e Copertura Ambi in corso...\n"); # Messaggio aggiornato
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    try:
        numero_target = entry_numero_target_simpatici.get().strip()
        if not numero_target or not numero_target.isdigit() or not (1 <= int(numero_target) <= 90):
            messagebox.showerror("Input Invalido", "Numero Target deve essere tra 1 e 90.")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Numero Target non valido.");risultato_text.config(state=tk.NORMAL);return
        
        selected_ruote_indices = listbox_ruote_simpatici.curselection(); nomi_ruote_selezionate_final = []
        all_ruote_in_listbox = [listbox_ruote_simpatici.get(i) for i in range(listbox_ruote_simpatici.size())]
        valid_ruote_from_listbox = [r for r in all_ruote_in_listbox if r not in ["Nessun file valido", "Nessun file ruota valido"]]
        
        if not selected_ruote_indices:
            if not valid_ruote_from_listbox:
                messagebox.showerror("Input Invalido", "Nessuna ruota valida disponibile per la ricerca.")
                risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Nessuna ruota valida disponibile.");risultato_text.config(state=tk.NORMAL);return
            nomi_ruote_selezionate_final = valid_ruote_from_listbox
        else:
            nomi_ruote_selezionate_final = [listbox_ruote_simpatici.get(i) for i in selected_ruote_indices]
        
        if not nomi_ruote_selezionate_final: # Doppio controllo se la lista è vuota
            messagebox.showerror("Input Invalido", "Selezionare almeno una ruota o assicurarsi che ci siano ruote valide.")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Nessuna ruota selezionata.");risultato_text.config(state=tk.NORMAL);return

        top_n_str = entry_top_n_simpatici.get().strip()
        if not top_n_str.isdigit() or int(top_n_str) <= 0:
            messagebox.showerror("Input Invalido", "Numero di 'Top N Simpatici' deve essere un intero positivo.")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Top N non valido.");risultato_text.config(state=tk.NORMAL);return
        top_n = int(top_n_str)
        
        start_dt = start_date_entry.get_date(); end_dt = end_date_entry.get_date()
        if start_dt > end_dt:
            messagebox.showerror("Input Date", "Data di inizio non può essere successiva alla data di fine.")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Date non valide.");risultato_text.config(state=tk.NORMAL);return
        start_ts = pd.Timestamp(start_dt); end_ts = pd.Timestamp(end_dt)
    except Exception as e:
        messagebox.showerror("Errore Input", f"Errore input: {e}")
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input: {e}");risultato_text.config(state=tk.NORMAL);return

    # MODIFICATO: La funzione ora restituisce più valori
    risultati_simpatici_list, errore_msg, occorrenze_target, risultati_copertura_ambi, _ = trova_abbinamenti_numero_target(
        numero_target, nomi_ruote_selezionate_final, start_ts, end_ts, top_n
        # max_k_ambi_copertura può essere passato se vuoi renderlo configurabile, altrimenti usa il default (es. 5)
    )

    output_lines = [f"=== Risultati Ricerca Numeri Simpatici & Copertura Ambi ==="] # Titolo aggiornato
    output_lines.append(f"Numero Target Analizzato: {numero_target.zfill(2)}")
    output_lines.append(f"Ruote Analizzate: {', '.join(nomi_ruote_selezionate_final) if nomi_ruote_selezionate_final else 'Nessuna ruota specificata o valida'}")
    output_lines.append(f"Periodo: {start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}")
    
    if errore_msg:
        output_lines.append(f"\nMessaggio dal sistema: {errore_msg}")
    
    if risultati_simpatici_list is None: # Errore grave nella funzione
        output_lines.append("Ricerca fallita o interrotta.")
    elif occorrenze_target == 0:
        if not errore_msg: # Se non c'è già un errore più specifico
            output_lines.append(f"\nIl Numero Target '{numero_target.zfill(2)}' non è stato trovato nel periodo e sulle ruote specificate.")
    else: # Il numero target è stato trovato almeno una volta
        output_lines.append(f"\nIl Numero Target '{numero_target.zfill(2)}' è stato trovato {occorrenze_target} volte.")
        
        if not risultati_simpatici_list:
            output_lines.append("Nessun altro numero abbinato trovato (nessun Numero Simpatico).")
        else:
            output_lines.append(f"\nTop {len(risultati_simpatici_list)} Numeri Simpatici (Numero: Frequenza Abbinamento):")
            for i, (num_simp, freq) in enumerate(risultati_simpatici_list):
                output_lines.append(f"  {i+1}. Numero {str(num_simp).zfill(2)}: {freq} volte")

        # --- NUOVA SEZIONE PER STAMPARE COPERTURA AMBI SIMPATICI ---
        if risultati_copertura_ambi:
            num_casi_target_per_copertura_ambi = len(date_uscite_target_con_ambi_effettivi) if 'date_uscite_target_con_ambi_effettivi' in locals() or 'date_uscite_target_con_ambi_effettivi' in globals() and date_uscite_target_con_ambi_effettivi else occorrenze_target
            
            output_lines.append(f"\n\n--- Copertura Combinata Ambi (Target '{numero_target.zfill(2)}' + Simpatico) ---")
            output_lines.append(f"(Analisi su {num_casi_target_per_copertura_ambi} uscite del Target con ambi validi)")
            if not risultati_copertura_ambi:
                 output_lines.append("  Nessuna combinazione di ambi simpatici ha coperto le uscite del target.")
            for item_cop_ambo in risultati_copertura_ambi:
                k_a = item_cop_ambo['k_ambi']
                ambi_list_str_a = item_cop_ambo['ambi_combinati']
                casi_cop_a = item_cop_ambo['casi_target_coperti']
                perc_cop_a = item_cop_ambo['percentuale_copertura']
                
                output_lines.append(f"\n  Combinazione di {k_a} Amb{'o' if k_a == 1 else 'i'} Simpatic{'o' if k_a == 1 else 'i'}:")
                for ambo_s in ambi_list_str_a:
                    output_lines.append(f"    - {ambo_s}")
                output_lines.append(f"    Ha coperto (almeno un ambo presente) {casi_cop_a} uscite del Target ({perc_cop_a:.1f}%)")
        elif risultati_simpatici_list: # Ci sono simpatici ma non risultati di copertura (es. il target non è mai uscito con altri numeri)
            output_lines.append("\n\n--- Copertura Combinata Ambi (Target + Simpatico) ---")
            output_lines.append("  Non è stato possibile calcolare la copertura combinata degli ambi (es. il Target è sempre uscito da solo o non ci sono dati sufficienti).")
        # --- FINE NUOVA SEZIONE ---

    final_output_str = "\n".join(output_lines)
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, final_output_str)
    risultato_text.see(tk.END)

    # Logica per decidere se mostrare il popup
    mostra_popup = False
    if errore_msg: # Se c'è un messaggio di errore, mostralo
        mostra_popup = True
    elif risultati_simpatici_list is not None and occorrenze_target > 0: # Se ci sono risultati validi
        mostra_popup = True
    
    if mostra_popup:
         mostra_popup_testo_semplice(f"Numeri Simpatici e Copertura Ambi per {numero_target.zfill(2)}", final_output_str)


# =============================================================================
# GUI e Mainloop (INVARIATE rispetto alla versione precedente,
# tranne il messaggio di benvenuto per riflettere il numero di colpi)
# =============================================================================
root = tk.Tk()
root.title("Numeri Spia - Ambo spia - Spia in posizione- Marker e Simpatici- Created by Il Lotto di Max")
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
input_params_simpatici_frame = ttk.Frame(controls_frame_simpatici_outer); input_params_simpatici_frame.grid(row=0, column=0, sticky="ns", padx=(0,10)); lbl_numero_target = ttk.Label(input_params_simpatici_frame, text="Numero Target (1-90):", style="Title.TLabel"); lbl_numero_target.pack(anchor="w", pady=(0,2))
entry_numero_target_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_numero_target_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_numero_target_simpatici.insert(0, "10"); lbl_top_n_simpatici = ttk.Label(input_params_simpatici_frame, text="Quanti Numeri Simpatici (Top N):", style="Title.TLabel"); lbl_top_n_simpatici.pack(anchor="w", pady=(5,2))
entry_top_n_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_top_n_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_top_n_simpatici.insert(0, "10"); button_cerca_simpatici = ttk.Button(input_params_simpatici_frame, text="Cerca Numeri Simpatici", command=esegui_ricerca_numeri_simpatici); button_cerca_simpatici.pack(pady=10, fill=tk.X, ipady=3)
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
post_analysis_checks_frame = ttk.Frame(main_frame); post_analysis_checks_frame.pack(fill=tk.X,pady=(5,0))
verifica_futura_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Predittiva (Post-Analisi) ",padding=5); verifica_futura_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y,expand=False)
ttk.Label(verifica_futura_frame,text=f"Controlla N Colpi (1-{MAX_COLPI_GIOCO}):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_futura = ttk.Entry(verifica_futura_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_futura.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_futura.insert(0,"9")
button_verifica_futura = ttk.Button(verifica_futura_frame,text="Verifica Futura\n(Post-Analisi)",command=esegui_verifica_futura); button_verifica_futura.pack(pady=5,fill=tk.X,ipady=0); button_verifica_futura.config(state=tk.DISABLED)
verifica_mista_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Mista (su Trigger Spia) ",padding=5); verifica_mista_frame.pack(side=tk.LEFT,padx=10,fill=tk.BOTH,expand=True)
ttk.Label(verifica_mista_frame, text="Combinazioni (1-5 numeri, 1-90, una per riga, separate da '-' o spazio):", style="Small.TLabel").pack(anchor="w")
text_mista_container = ttk.Frame(verifica_mista_frame); text_mista_container.pack(fill=tk.BOTH, expand=True, pady=(0,5)); text_combinazioni_miste_scrollbar_y = ttk.Scrollbar(text_mista_container, orient=tk.VERTICAL); text_combinazioni_miste_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_combinazioni_miste = tk.Text(text_mista_container, height=4, width=30, font=("Consolas", 10), wrap=tk.WORD, yscrollcommand=text_combinazioni_miste_scrollbar_y.set, bd=1, relief="sunken"); text_combinazioni_miste.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); text_combinazioni_miste_scrollbar_y.config(command=text_combinazioni_miste.yview); text_combinazioni_miste.insert("1.0", "18 23 43\n60")
ttk.Label(verifica_mista_frame, text=f"Controlla N Colpi Verifica (1-{MAX_COLPI_GIOCO}):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_mista = ttk.Entry(verifica_mista_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_mista.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_mista.insert(0,"9")
button_verifica_mista = ttk.Button(verifica_mista_frame,text="Verifica Mista\n(su Trigger Spia)",command=esegui_verifica_mista); button_verifica_mista.pack(pady=5,fill=tk.X,ipady=0); button_verifica_mista.config(state=tk.DISABLED)
ttk.Label(main_frame, text="Risultati Analisi (Log):", style="Header.TLabel").pack(anchor="w",pady=(15,0))
risultato_outer_frame = ttk.Frame(main_frame); risultato_outer_frame.pack(fill=tk.BOTH,expand=True,pady=5); risultato_scroll_y = ttk.Scrollbar(risultato_outer_frame,orient=tk.VERTICAL); risultato_scroll_y.pack(side=tk.RIGHT,fill=tk.Y); risultato_scroll_x = ttk.Scrollbar(risultato_outer_frame,orient=tk.HORIZONTAL); risultato_scroll_x.pack(side=tk.BOTTOM,fill=tk.X)
risultato_text = tk.Text(risultato_outer_frame,wrap=tk.NONE,font=("Consolas",10),height=15,yscrollcommand=risultato_scroll_y.set,xscrollcommand=risultato_scroll_x.set,state=tk.NORMAL,bd=1,relief="sunken")
risultato_text.pack(fill=tk.BOTH,expand=True); risultato_scroll_y.config(command=risultato_text.yview); risultato_scroll_x.config(command=risultato_text.xview)

def aggiorna_lista_file_gui(target_listbox):
    global file_ruote
    if not target_listbox: return
    target_listbox.config(state=tk.NORMAL); target_listbox.delete(0, tk.END)

    # Modifica per ordinamento personalizzato: NAZIONALE per ultima
    lista_ruote_originale = list(file_ruote.keys())
    ruota_nazionale_str = "NAZIONALE"

    ruote_ordinate = sorted([r for r in lista_ruote_originale if r != ruota_nazionale_str])

    if ruota_nazionale_str in lista_ruote_originale:
        ruote_ordinate.append(ruota_nazionale_str)
    # Fine modifica

    if ruote_ordinate:
        for r in ruote_ordinate:
            target_listbox.insert(tk.END, r)
    else:
        target_listbox.insert(tk.END, "Nessun file ruota valido")
        target_listbox.config(state=tk.DISABLED)

def mappa_file_ruote():
    global file_ruote, cartella_entry
    cartella = cartella_entry.get(); file_ruote.clear()
    if not cartella or not os.path.isdir(cartella): return False
    ruote_valide = ['BARI','CAGLIARI','FIRENZE','GENOVA','MILANO','NAPOLI','PALERMO','ROMA','TORINO','VENEZIA','NAZIONALE']; found = False
    try:
        for file in os.listdir(cartella):
            fp = os.path.join(cartella, file)
            if os.path.isfile(fp) and file.lower().endswith(".txt"):
                nome_base = os.path.splitext(file)[0].upper()
                if nome_base in ruote_valide: file_ruote[nome_base] = fp; found = True
        return found
    except OSError as e: print(f"Errore lettura cartella: {e}"); return False
    except Exception as e: print(f"Errore scansione file: {e}"); traceback.print_exc(); return False

def on_sfoglia_click():
    global cartella_entry, listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici
    cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni")
    if cartella_sel:
        cartella_entry.delete(0,tk.END); cartella_entry.insert(0,cartella_sel)
        if mappa_file_ruote(): _=[aggiorna_lista_file_gui(lb) for lb in [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici] if lb]
        else: _=[(lb.config(state=tk.NORMAL),lb.delete(0,tk.END),lb.insert(tk.END,"Nessun file valido"),lb.config(state=tk.DISABLED)) for lb in [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici] if lb]; messagebox.showwarning("Nessun File", "Nessun file .txt valido trovato.")
btn_sfoglia.config(command=on_sfoglia_click)

def main():
    global root, risultato_text, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    welcome_message = (
        f"Benvenuto in Numeri Spia (Colpi Max: {MAX_COLPI_GIOCO}, Esito 'IN CORSO', Copia/Salva Popup)!\n\n"
        "1. Usa 'Sfoglia...' per selezionare la cartella delle estrazioni.\n"
        "2. Imposta il periodo di analisi.\n"
        "3. Scegli la modalità di analisi:\n"
        "   - 'Analisi Numeri Successivi (Spia)':\n"
        "     - Ruote Analisi (spia) e Verifica (esiti).\n"
        "     - Tipo Spia (Estratto, Ambo, Estratto Posizionale).\n"
        f"     - Estrazioni Successive da controllare (1-{MAX_COLPI_GIOCO}).\n"
        "     - Clicca 'Cerca Successivi'.\n"
        "   - 'Analisi Numeri Antecedenti (Marker)': (Dettagli nella tab)\n"
        "   - 'Numeri Simpatici': (Dettagli nella tab)\n"
        f"\nFunzioni di Verifica (usano i risultati dell'analisi 'Successivi'):\n"
        f" - 'Verifica Predittiva': controlla esiti per N colpi (1-{MAX_COLPI_GIOCO}) dopo la data fine analisi.\n"
        f" - 'Verifica Mista': controlla combinazioni personalizzate per N colpi (1-{MAX_COLPI_GIOCO}) dopo ogni trigger spia.\n"
        "   (NOVITÀ) Esito 'IN CORSO' se i dati finiscono prima dei colpi richiesti.\n"
        "\nI risultati nel box sottostante sono copiabili. I popup hanno un pulsante 'Salva su File...'."
    )
    risultato_text.insert(tk.END, welcome_message)
    root.mainloop()
    print("\nScript terminato.")

if __name__ == "__main__":
    main()