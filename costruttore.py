import os
from datetime import datetime, date
from collections import Counter, defaultdict
from itertools import combinations
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext
try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Errore Dipendenza", "Il pacchetto 'tkcalendar' non è installato.\nPer favore, installalo con: pip install tkcalendar")
    sys.exit()

# --- COSTANTI GLOBALI ---
RUOTE = ["Bari", "Cagliari", "Firenze", "Genova", "Milano", "Napoli", "Palermo", "Roma", "Torino", "Venezia", "Nazionale"]
SIGLE_RUOTE_MAP = {
    "Bari": "BA", "Cagliari": "CA", "Firenze": "FI", "Genova": "GE",
    "Milano": "MI", "Napoli": "NA", "Palermo": "PA", "Roma": "RM",
    "Torino": "TO", "Venezia": "VE", "Nazionale": "RN"
}
OPERAZIONI = {
    "somma": lambda a, b: a + b,
    "differenza": lambda a, b: a - b,
    "moltiplicazione": lambda a, b: a * b,
}

# --- FUNZIONI LOGICHE (ora prenderanno `app_logger` come argomento se devono loggare) ---
def regola_fuori_90(numero):
    if numero == 0: return 90
    if numero < 0:
        abs_val = abs(numero); resto = abs_val % 90
        return 90 - resto if resto != 0 else 90
    return (numero - 1) % 90 + 1

def parse_riga_estrazione(riga, nome_file_ruota, num_riga):
    try:
        parti = riga.strip().split()
        if len(parti) < 7: return None, None
        data_str = parti[0]; numeri_str = parti[2:7]
        numeri = sorted([int(n) for n in numeri_str])
        if len(numeri) != 5: return None, None
        data_obj = datetime.strptime(data_str, "%Y/%m/%d").date()
        return data_obj, numeri
    except: return None, None

def carica_storico_completo(cartella_dati, data_inizio_filtro=None, data_fine_filtro=None, app_logger=None):
    def log_message(msg):
        if app_logger: app_logger(msg)
        else: print(msg)
    # ... (resto della funzione come prima, usando log_message) ...
    log_message(f"\nCaricamento dati da: {cartella_dati}")
    if data_inizio_filtro or data_fine_filtro: log_message(f"Filtro date: Da {data_inizio_filtro or 'inizio'} a {data_fine_filtro or 'fine'}")
    if not os.path.isdir(cartella_dati): log_message(f"Errore: Cartella '{cartella_dati}' non trovata."); return []
    storico_globale = defaultdict(dict)
    file_trovati_cont = 0; righe_valide_tot = 0; righe_processate_tot = 0
    for nome_ruota_chiave in RUOTE:
        nome_file_da_cercare = f"{nome_ruota_chiave.upper()}.TXT"
        path_file = os.path.join(cartella_dati, nome_file_da_cercare)
        if not os.path.exists(path_file):
            path_file_fallback = os.path.join(cartella_dati, f"{nome_ruota_chiave}.txt")
            if os.path.exists(path_file_fallback): path_file = path_file_fallback
            else: continue
        file_trovati_cont += 1; righe_nel_file = 0; righe_valide_nel_file = 0
        try:
            with open(path_file, 'r', encoding='utf-8') as f:
                for num_riga, riga_contenuto in enumerate(f, 1):
                    righe_nel_file += 1
                    if not riga_contenuto.strip(): continue
                    data_obj, numeri = parse_riga_estrazione(riga_contenuto, os.path.basename(path_file), num_riga)
                    if data_obj and numeri:
                        if data_inizio_filtro and data_obj < data_inizio_filtro: continue
                        if data_fine_filtro and data_obj > data_fine_filtro: continue
                        righe_valide_nel_file +=1
                        if nome_ruota_chiave in storico_globale[data_obj] and storico_globale[data_obj][nome_ruota_chiave] != numeri:
                            log_message(f"Attenzione: Dati discordanti per {nome_ruota_chiave} il {data_obj}.")
                        elif nome_ruota_chiave not in storico_globale[data_obj]:
                             storico_globale[data_obj][nome_ruota_chiave] = numeri
            righe_processate_tot += righe_nel_file; righe_valide_tot += righe_valide_nel_file
        except Exception as e: log_message(f"Errore grave leggendo {path_file}: {e}")
    if file_trovati_cont == 0: log_message("Nessun file archivio valido trovato."); return []
    log_message(f"Processate {righe_valide_tot}/{righe_processate_tot} righe valide da {file_trovati_cont} file.")
    storico_ordinato = []
    date_ordinate = sorted(storico_globale.keys())
    estrazioni_mese_corrente = 0; mese_precedente, anno_precedente = None, None
    for data_obj in date_ordinate:
        if anno_precedente != data_obj.year or mese_precedente != data_obj.month:
            estrazioni_mese_corrente = 1; mese_precedente, anno_precedente = data_obj.month, data_obj.year
        else: estrazioni_mese_corrente += 1
        estrazione_completa = {'data': data_obj, 'indice_mese': estrazioni_mese_corrente}
        for r_nome in RUOTE: estrazione_completa[r_nome] = storico_globale[data_obj].get(r_nome, [])
        if any(estrazione_completa[r_n] for r_n in RUOTE): storico_ordinato.append(estrazione_completa)
    log_message(f"Caricate e ordinate {len(storico_ordinato)} estrazioni complessive valide.")
    return storico_ordinato


def analizza_metodo_sommativo_base(storico, ruota_calcolo, pos_estratto_calcolo, operazione_str, operando_fisso, ruote_gioco_selezionate, lookahead=1, indice_mese_filtro=None):
    # ... (Logica interna invariata, non necessita di app_logger direttamente) ...
    if operazione_str not in OPERAZIONI: raise ValueError(f"Operazione '{operazione_str}' non supportata.")
    op_func = OPERAZIONI[operazione_str]
    successi, tentativi = 0, 0; applicazioni_vincenti = []
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
        if not estrazione_corrente.get(ruota_calcolo) or len(estrazione_corrente[ruota_calcolo]) <= pos_estratto_calcolo: continue
        numero_base = estrazione_corrente[ruota_calcolo][pos_estratto_calcolo]
        try: valore_operazione = op_func(numero_base, operando_fisso)
        except ZeroDivisionError: continue
        ambata_prevista = regola_fuori_90(valore_operazione); tentativi += 1; trovato_in_questo_tentativo = False; dettagli_vincita_per_tentativo = []
        for k in range(1, lookahead + 1):
            if i + k >= len(storico): break
            estrazione_futura = storico[i + k]
            for ruota_verifica_effettiva in ruote_gioco_selezionate:
                if ambata_prevista in estrazione_futura.get(ruota_verifica_effettiva, []):
                    if not trovato_in_questo_tentativo: successi += 1; trovato_in_questo_tentativo = True
                    dettagli_vincita_per_tentativo.append({"ruota_vincita": ruota_verifica_effettiva, "numeri_ruota_vincita": estrazione_futura.get(ruota_verifica_effettiva, []), "data_riscontro": estrazione_futura['data'], "colpo_riscontro": k})
            if trovato_in_questo_tentativo and len(ruote_gioco_selezionate) == 1: break
        if trovato_in_questo_tentativo: applicazioni_vincenti.append({"data_applicazione": estrazione_corrente['data'], "estratto_base": numero_base, "operando": operando_fisso, "operazione": operazione_str, "ambata_prevista": ambata_prevista, "riscontri": dettagli_vincita_per_tentativo})
    return ambata_prevista, successi, tentativi, applicazioni_vincenti

def trova_migliori_ambate_e_abbinamenti(storico, ruota_calcolo, pos_estratto_calcolo, ruote_gioco_selezionate, max_ambate_output=1, lookahead=1, indice_mese_filtro=None, min_tentativi_per_ambata=10, app_logger=None):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)
        else: print(msg, end=end, flush=flush)
    # ... (resto della funzione come prima, usando log_message) ...
    risultati_ambate = []; gioco_su_desc = "su " + ", ".join(ruote_gioco_selezionate) if len(ruote_gioco_selezionate) < len(RUOTE) else "su TUTTE le ruote"
    log_message(f"\nAnalisi metodi per ambata {gioco_su_desc} (da {ruota_calcolo}[pos.{pos_estratto_calcolo+1}]):")
    tot_metodi_testati = len(OPERAZIONI) * 90; metodi_processati = 0
    for op_str in OPERAZIONI:
        for operando in range(1, 91):
            metodi_processati += 1
            if metodi_processati % 10 == 0 or metodi_processati == tot_metodi_testati : log_message(f"  Testando metodo {metodi_processati}/{tot_metodi_testati}...", end='\r', flush=True)
            _, successi, tentativi, applicazioni_vincenti_dett = analizza_metodo_sommativo_base(storico, ruota_calcolo, pos_estratto_calcolo, op_str, operando, ruote_gioco_selezionate, lookahead, indice_mese_filtro)
            if tentativi >= min_tentativi_per_ambata:
                frequenza = successi / tentativi if tentativi > 0 else 0
                risultati_ambate.append({"metodo": {"operazione": op_str, "operando_fisso": operando}, "successi": successi, "tentativi": tentativi, "frequenza_ambata": frequenza, "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dett})
    log_message("\n  Completata analisi metodi per ambata.                                  ")
    risultati_ambate.sort(key=lambda x: (x["frequenza_ambata"], x["successi"]), reverse=True)
    top_ambate_con_abbinamenti = []; log_message(f"\nAnalisi abbinamenti per le top {min(max_ambate_output, len(risultati_ambate))} ambate...")
    for i, res_ambata in enumerate(risultati_ambate[:max_ambate_output]):
        log_message(f"  Analizzando abbinamenti per il metodo {i+1} ({res_ambata['metodo']['operazione']} {res_ambata['metodo']['operando_fisso']})...")
        if not res_ambata["applicazioni_vincenti_dettagliate"]: log_message("    Nessuna applicazione vincente."); continue
        conta_ambate_specifiche_da_metodo = Counter()
        for app_vinc in res_ambata["applicazioni_vincenti_dettagliate"]: conta_ambate_specifiche_da_metodo[app_vinc['ambata_prevista']] +=1
        if not conta_ambate_specifiche_da_metodo: log_message("    Metodo non ha prodotto ambate vincenti consistenti."); continue
        ambata_target_per_abbinamenti = conta_ambate_specifiche_da_metodo.most_common(1)[0][0]
        abbinamenti_per_ambo = Counter(); abbinamenti_per_terno = Counter(); abbinamenti_per_quaterna = Counter(); abbinamenti_per_cinquina = Counter()
        conteggio_eventi_per_abbinamenti = 0
        for app_vinc in res_ambata["applicazioni_vincenti_dettagliate"]:
            if app_vinc['ambata_prevista'] != ambata_target_per_abbinamenti: continue
            for riscontro_info in app_vinc["riscontri"]:
                conteggio_eventi_per_abbinamenti +=1
                numeri_usciti_su_ruota_vincita = [n for n in riscontro_info["numeri_ruota_vincita"] if n != ambata_target_per_abbinamenti]
                for num_abbinato in numeri_usciti_su_ruota_vincita: abbinamenti_per_ambo[num_abbinato] += 1
                if len(numeri_usciti_su_ruota_vincita) >= 2:
                    for combo_2 in combinations(sorted(numeri_usciti_su_ruota_vincita), 2): abbinamenti_per_terno[combo_2] += 1
                if len(numeri_usciti_su_ruota_vincita) >= 3:
                    for combo_3 in combinations(sorted(numeri_usciti_su_ruota_vincita), 3): abbinamenti_per_quaterna[combo_3] += 1
                if len(numeri_usciti_su_ruota_vincita) >= 4:
                    for combo_4 in combinations(sorted(numeri_usciti_su_ruota_vincita), 4): abbinamenti_per_cinquina[combo_4] += 1
        res_ambata["ambata_piu_frequente_dal_metodo"] = ambata_target_per_abbinamenti
        res_ambata["abbinamenti"] = {"ambo": [{"numeri": [ab[0]], "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_ambo.most_common(5)], "terno": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_terno.most_common(5)], "quaterna": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_quaterna.most_common(5)], "cinquina": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_cinquina.most_common(5)], "eventi_abbinamento_analizzati": conteggio_eventi_per_abbinamenti}
        if "applicazioni_vincenti_dettagliate" in res_ambata: del res_ambata["applicazioni_vincenti_dettagliate"]
        top_ambate_con_abbinamenti.append(res_ambata)
    log_message("  Completata analisi abbinamenti.")
    return top_ambate_con_abbinamenti


def verifica_giocata_manuale(numeri_da_giocare, ruote_selezionate, data_inizio_controllo,
                             num_colpi_controllo, storico_completo, app_logger=None):
    def log_message(msg): # Logger locale per questa funzione
        if app_logger: app_logger(msg)
        else: print(msg)
    # ... (resto della funzione come prima, usando log_message) ...
    log_message(f"\n--- VERIFICA GIOCATA MANUALE ---")
    log_message(f"Numeri da giocare: {numeri_da_giocare}")
    log_message(f"Ruote selezionate: {', '.join(ruote_selezionate)}")
    log_message(f"Data inizio controllo: {data_inizio_controllo}")
    log_message(f"Numero colpi controllo: {num_colpi_controllo}")
    if not numeri_da_giocare: log_message("ERRORE: Nessun numero inserito per la verifica."); return
    if not ruote_selezionate: log_message("ERRORE: Nessuna ruota selezionata per la verifica."); return
    if not data_inizio_controllo: log_message("ERRORE: Data inizio controllo non specificata."); return
    indice_partenza = -1
    for i, estrazione in enumerate(storico_completo):
        if estrazione['data'] >= data_inizio_controllo: indice_partenza = i; break
    if indice_partenza == -1: log_message(f"Nessuna estrazione trovata a partire dal {data_inizio_controllo}."); return
    log_message(f"Controllo a partire dall'estrazione del {storico_completo[indice_partenza]['data']}:")
    trovato_esito = False
    for colpo in range(num_colpi_controllo):
        indice_estrazione_corrente = indice_partenza + colpo
        if indice_estrazione_corrente >= len(storico_completo):
            log_message(f"Fine storico raggiunto prima di completare {num_colpi_controllo} colpi."); break
        estrazione_controllo = storico_completo[indice_estrazione_corrente]
        log_message(f"  Colpo {colpo + 1} (Data: {estrazione_controllo['data']}):")
        for ruota in ruote_selezionate:
            numeri_estratti_ruota = estrazione_controllo.get(ruota, [])
            if not numeri_estratti_ruota: continue
            vincenti_ambata = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
            if vincenti_ambata: log_message(f"    >> AMBATA SU {ruota.upper()}! Numeri usciti: {vincenti_ambata}"); trovato_esito = True
            if len(numeri_da_giocare) >= 2:
                numeri_giocati_presenti_nella_ruota = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
                num_corrispondenze = len(numeri_giocati_presenti_nella_ruota)
                if num_corrispondenze == 2: log_message(f"    >> AMBO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"); trovato_esito = True
                elif num_corrispondenze == 3: log_message(f"    >> TERNO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"); trovato_esito = True
                elif num_corrispondenze == 4: log_message(f"    >> QUATERNA SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"); trovato_esito = True
                elif num_corrispondenze == 5 and len(numeri_da_giocare) >= 5: log_message(f"    >> CINQUINA SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"); trovato_esito = True
        if trovato_esito and len(ruote_selezionate) == 1: log_message(f"--- Esito trovato al colpo {colpo + 1} ---"); break
    if not trovato_esito: log_message(f"\nNessun esito trovato per i numeri {numeri_da_giocare} entro {num_colpi_controllo} colpi.")
    log_message("--- Fine Verifica Giocata Manuale ---")


# --- CLASSE PER LA GUI ---
class LottoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Costruttore Metodi Lotto Avanzato")
        master.geometry("750x850")

        # ... (Variabili Tkinter come prima) ...
        self.cartella_dati_var = tk.StringVar()
        self.ruota_calcolo_var = tk.StringVar(value=RUOTE[0])
        self.posizione_estratto_var = tk.IntVar(value=1)
        self.ruote_gioco_vars = {ruota: tk.BooleanVar() for ruota in RUOTE}
        self.tutte_le_ruote_var = tk.BooleanVar(value=True)
        self.lookahead_var = tk.IntVar(value=3)
        self.indice_mese_var = tk.StringVar()
        self.num_ambate_var = tk.IntVar(value=1)
        self.min_tentativi_var = tk.IntVar(value=10)
        self.numeri_verifica_var = tk.StringVar()
        self.colpi_verifica_var = tk.IntVar(value=9)
        self.storico_caricato_per_verifica = None # Cache

        # Frame per i controlli di input principali (analisi)
        main_input_frame = ttk.LabelFrame(master, text="Impostazioni Analisi Metodi", padding="10")
        main_input_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)
        current_row = 0
        # ... (Layout dei widget di input per ANALISI come prima) ...
        tk.Label(main_input_frame, text="Cartella Archivio Dati:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(main_input_frame, textvariable=self.cartella_dati_var, width=50).grid(row=current_row, column=1, sticky="ew", padx=5, pady=2)
        tk.Button(main_input_frame, text="Sfoglia...", command=self.seleziona_cartella).grid(row=current_row, column=2, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Data Inizio Analisi:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        self.date_inizio_entry_analisi = DateEntry(main_input_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_inizio_entry_analisi.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        tk.Button(main_input_frame, text="Nessuna", command=lambda: self.date_inizio_entry_analisi.delete(0, tk.END)).grid(row=current_row, column=2, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Data Fine Analisi:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        self.date_fine_entry_analisi = DateEntry(main_input_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_fine_entry_analisi.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        tk.Button(main_input_frame, text="Nessuna", command=lambda: self.date_fine_entry_analisi.delete(0, tk.END)).grid(row=current_row, column=2, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Ruota Calcolo:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(main_input_frame, textvariable=self.ruota_calcolo_var, values=RUOTE, state="readonly", width=15).grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Posizione Estratto (1-5):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(main_input_frame, from_=1, to=5, textvariable=self.posizione_estratto_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Ruote Gioco (per Analisi):").grid(row=current_row, column=0, sticky="nw", padx=5, pady=2)
        ruote_frame_analisi = tk.Frame(main_input_frame)
        ruote_frame_analisi.grid(row=current_row, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        tk.Checkbutton(ruote_frame_analisi, text="Tutte le Ruote", variable=self.tutte_le_ruote_var, command=self.toggle_tutte_ruote).grid(row=0, column=0, columnspan=4, sticky="w")
        self.check_ruote_gioco_widgets = []
        for i, ruota in enumerate(RUOTE):
            cb = tk.Checkbutton(ruote_frame_analisi, text=ruota, variable=self.ruote_gioco_vars[ruota], command=self.update_tutte_le_ruote_status)
            cb.grid(row=1 + i // 4, column=i % 4, sticky="w")
            self.check_ruote_gioco_widgets.append(cb)
        self.toggle_tutte_ruote()
        current_row += (len(RUOTE) // 4) + 2
        tk.Label(main_input_frame, text="Colpi di Gioco (Analisi):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(main_input_frame, from_=1, to=18, textvariable=self.lookahead_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Indice Mese (vuoto=tutte):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(main_input_frame, textvariable=self.indice_mese_var, width=7).grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="N. Ambate da Dettagliare:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(main_input_frame, from_=1, to=10, textvariable=self.num_ambate_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(main_input_frame, text="Min. Tentativi per Metodo:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(main_input_frame, from_=1, to=100, textvariable=self.min_tentativi_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Button(main_input_frame, text="Avvia Analisi Metodi", command=self.avvia_analisi, font=("Helvetica", 11, "bold"), bg="lightgreen").grid(row=current_row, column=0, columnspan=3, pady=10, ipady=3)

        # Frame per i controlli di Verifica Giocata Manuale
        verifica_frame = ttk.LabelFrame(master, text="Verifica Giocata Manuale", padding="10")
        verifica_frame.pack(padx=10, pady=10, fill=tk.X, expand=False)
        current_row_vf = 0
        tk.Label(verifica_frame, text="Numeri da Verificare (es. 23 o 23,45,67):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(verifica_frame, textvariable=self.numeri_verifica_var, width=30).grid(row=current_row_vf, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_frame, text="Data Inizio Verifica:").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        self.date_inizio_verifica_entry = DateEntry(verifica_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_inizio_verifica_entry.grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_frame, text="Colpi per Verifica (1-18):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(verifica_frame, from_=1, to=18, textvariable=self.colpi_verifica_var, width=5, state="readonly").grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_frame, text="Ruote Gioco (per Verifica): (Usa selezione da 'Impostazioni Analisi Metodi')").grid(row=current_row_vf, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Button(verifica_frame, text="Verifica Giocata", command=self.avvia_verifica_giocata, font=("Helvetica", 11, "bold"), bg="lightblue").grid(row=current_row_vf, column=0, columnspan=3, pady=10, ipady=3)

        # Area di Output Testuale
        output_label = tk.Label(master, text="Log e Risultati:", font=("Helvetica", 10, "bold"))
        output_label.pack(pady=(5,0), anchor="w", padx=10)
        self.output_text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=15, font=("Courier New", 9))
        self.output_text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.output_text_area.config(state=tk.DISABLED)


    # METODO _log_to_gui ORA PARTE DELLA CLASSE
    def _log_to_gui(self, message, end='\n', flush=False):
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.insert(tk.END, message + end)
        if flush:
            self.output_text_area.see(tk.END)
            self.output_text_area.update_idletasks()
        self.output_text_area.config(state=tk.DISABLED)

    def seleziona_cartella(self): # Invariato
        cartella = filedialog.askdirectory(title="Seleziona cartella archivi")
        if cartella: self.cartella_dati_var.set(cartella)

    def toggle_tutte_ruote(self): # Invariato
        stato_tutte = self.tutte_le_ruote_var.get()
        for ruota_var in self.ruote_gioco_vars.values(): ruota_var.set(stato_tutte)
        for cb_widget in self.check_ruote_gioco_widgets: cb_widget.config(state=tk.DISABLED if stato_tutte else tk.NORMAL)

    def update_tutte_le_ruote_status(self): # Invariato
        if all(var.get() for var in self.ruote_gioco_vars.values()): self.tutte_le_ruote_var.set(True)
        else: self.tutte_le_ruote_var.set(False)
        
    def _clear_output_area(self): # Invariato
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.delete('1.0', tk.END)
        self.output_text_area.config(state=tk.DISABLED)

    def avvia_analisi(self):
        self._clear_output_area()
        self._log_to_gui("\n" + "="*50 + "\nAVVIO ANALISI METODI\n" + "="*50)
        # ... (Raccogli parametri come prima) ...
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati):
            messagebox.showerror("Errore Input", "Seleziona una cartella archivio dati valida.")
            self._log_to_gui("ERRORE: Cartella dati non valida.") # USA self._log_to_gui
            return
        try: data_inizio_analisi = self.date_inizio_entry_analisi.get_date()
        except: data_inizio_analisi = None
        try: data_fine_analisi = self.date_fine_entry_analisi.get_date()
        except: data_fine_analisi = None
        if data_inizio_analisi and data_fine_analisi and data_fine_analisi < data_inizio_analisi:
            messagebox.showerror("Errore Input", "Data fine < data inizio per Analisi."); self._log_to_gui("ERRORE: Data fine < data inizio per Analisi."); return
        ruota_calcolo = self.ruota_calcolo_var.get()
        posizione_estratto = self.posizione_estratto_var.get() - 1
        ruote_gioco_selezionate_analisi = [ruota for ruota, var in self.ruote_gioco_vars.items() if var.get()]
        if self.tutte_le_ruote_var.get() or not ruote_gioco_selezionate_analisi:
            ruote_gioco_selezionate_analisi = RUOTE[:]
        if not ruote_gioco_selezionate_analisi:
            messagebox.showerror("Errore Input", "Seleziona almeno una ruota di gioco per l'Analisi."); self._log_to_gui("ERRORE: Nessuna ruota di gioco per l'Analisi."); return
        lookahead = self.lookahead_var.get()
        indice_mese_str = self.indice_mese_var.get(); indice_mese = None
        if indice_mese_str:
            try:
                indice_mese = int(indice_mese_str)
                if indice_mese <= 0: messagebox.showerror("Errore Input", "Indice mese deve essere positivo."); self._log_to_gui("ERRORE: Indice mese non positivo."); return
            except ValueError: messagebox.showerror("Errore Input", "Indice mese deve essere un numero."); self._log_to_gui("ERRORE: Indice mese non numerico."); return
        num_ambate = self.num_ambate_var.get(); min_tentativi = self.min_tentativi_var.get()

        self._log_to_gui(f"Parametri Analisi:\n  Cartella: {cartella_dati}\n  Date: {data_inizio_analisi} a {data_fine_analisi}\n  Da: {ruota_calcolo}[{posizione_estratto+1}]\n  Su: {', '.join(ruote_gioco_selezionate_analisi)}\n  Colpi: {lookahead}, Ind.Mese: {indice_mese}\n  OutAmbate: {num_ambate}, MinTent: {min_tentativi}")
        
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            self.storico_caricato_per_verifica = carica_storico_completo(cartella_dati, data_inizio_analisi, data_fine_analisi, app_logger=self._log_to_gui) # Passa il logger
            if not self.storico_caricato_per_verifica:
                messagebox.showinfo("Risultato Analisi", "Nessun dato storico caricato/filtrato.")
                self._log_to_gui("Nessun dato storico caricato/filtrato per l'analisi.")
                return

            risultati = trova_migliori_ambate_e_abbinamenti(
                self.storico_caricato_per_verifica, ruota_calcolo, posizione_estratto, ruote_gioco_selezionate_analisi,
                max_ambate_output=num_ambate, lookahead=lookahead,
                indice_mese_filtro=indice_mese, min_tentativi_per_ambata=min_tentativi,
                app_logger=self._log_to_gui # Passa il logger
            )
            self._log_to_gui("\n\n--- RISULTATI FINALI DELL'ANALISI ---")
            if not risultati: self._log_to_gui("Nessun metodo ha prodotto risultati sufficientemente frequenti.")
            else:
                for i, res in enumerate(risultati): # Stampa risultati usando self._log_to_gui
                    metodo = res['metodo']
                    self._log_to_gui(f"\n--- {i+1}° MIGLIOR METODO PER AMBATA ---")
                    self._log_to_gui(f"  Metodo: {ruota_calcolo}[pos.{posizione_estratto+1}] {metodo['operazione']} {metodo['operando_fisso']}")
                    self._log_to_gui(f"  Ambata più frequente prodotta: {res.get('ambata_piu_frequente_dal_metodo', 'N/D')}")
                    self._log_to_gui(f"  Frequenza successo Ambata (metodo): {res['frequenza_ambata']:.2%} ({res['successi']}/{res['tentativi']} casi)")
                    desc_ruote_gioco = "TUTTE le ruote" if len(ruote_gioco_selezionate_analisi) == len(RUOTE) else ", ".join(ruote_gioco_selezionate_analisi)
                    if len(ruote_gioco_selezionate_analisi) > 1 : self._log_to_gui(f"    (Conteggio successi su: {desc_ruote_gioco})")
                    abbinamenti = res.get("abbinamenti", {}); eventi_abbinamento = abbinamenti.get("eventi_abbinamento_analizzati", 0)
                    if eventi_abbinamento > 0:
                        self._log_to_gui(f"  Analizzati {eventi_abbinamento} eventi di vincita per abbinamenti con ambata '{res.get('ambata_piu_frequente_dal_metodo', 'N/D')}':")
                        for tipo_sorte, dati_sorte_lista in abbinamenti.items():
                            if tipo_sorte == "eventi_abbinamento_analizzati": continue
                            if dati_sorte_lista:
                                self._log_to_gui(f"    Migliori Abbinamenti per {tipo_sorte.upper().replace('_', ' ')}:")
                                mostrati = 0
                                for ab_info in dati_sorte_lista:
                                    if ab_info['conteggio'] > 0:
                                        numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                        self._log_to_gui(f"      - Numeri abbinati: [{numeri_ab_str}] -> Freq. {ab_info['frequenza']:.2%} (Conteggio: {ab_info['conteggio']})")
                                        mostrati +=1
                                if mostrati == 0: self._log_to_gui(f"      Nessun abbinamento significativo per {tipo_sorte.upper()}.")
                    else: self._log_to_gui(f"  Nessun caso di successo del metodo ha prodotto l'ambata target '{res.get('ambata_piu_frequente_dal_metodo', 'N/D')}' per analizzare abbinamenti.")

            self._log_to_gui("\n--- Analisi Metodi Completata ---")
            messagebox.showinfo("Analisi Completata", "Analisi Metodi terminata. Vedi risultati nell'area sottostante.")
        except Exception as e:
            messagebox.showerror("Errore Analisi", f"Errore: {e}"); self._log_to_gui(f"ERRORE ANALISI: {e}")
            import traceback; self._log_to_gui(traceback.format_exc())
        finally: self.master.config(cursor="")

    def avvia_verifica_giocata(self):
        self._clear_output_area()
        self._log_to_gui("\n" + "="*50 + "\nAVVIO VERIFICA GIOCATA MANUALE\n" + "="*50)
        # ... (Raccogli parametri per la verifica come prima) ...
        numeri_str = self.numeri_verifica_var.get()
        try:
            numeri_da_verificare = sorted([int(n.strip()) for n in numeri_str.split(',') if n.strip()])
            if not numeri_da_verificare: messagebox.showerror("Errore Input", "Inserisci almeno un numero da verificare."); self._log_to_gui("ERRORE: Nessun numero da verificare."); return
            if not all(1 <= n <= 90 for n in numeri_da_verificare): messagebox.showerror("Errore Input", "Numeri devono essere tra 1 e 90."); self._log_to_gui("ERRORE: Numeri non validi."); return
        except ValueError: messagebox.showerror("Errore Input", "Formato numeri non valido."); self._log_to_gui("ERRORE: Formato numeri non valido."); return
        try: data_inizio_ver = self.date_inizio_verifica_entry.get_date()
        except: messagebox.showerror("Errore Input", "Seleziona data inizio verifica."); self._log_to_gui("ERRORE: Data inizio verifica non selezionata."); return
        colpi_ver = self.colpi_verifica_var.get()
        ruote_gioco_selezionate_ver = [ruota for ruota, var in self.ruote_gioco_vars.items() if var.get()]
        if self.tutte_le_ruote_var.get() or not ruote_gioco_selezionate_ver: ruote_gioco_selezionate_ver = RUOTE[:]
        if not ruote_gioco_selezionate_ver: messagebox.showerror("Errore Input", "Seleziona ruote gioco per Verifica."); self._log_to_gui("ERRORE: Nessuna ruota di gioco per Verifica."); return
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati): messagebox.showerror("Errore Input", "Seleziona cartella archivio dati."); self._log_to_gui("ERRORE: Cartella dati non valida."); return

        self._log_to_gui(f"Parametri Verifica:\n  Numeri: {numeri_da_verificare}\n  Data Inizio: {data_inizio_ver}\n  Colpi: {colpi_ver}\n  Ruote: {', '.join(ruote_gioco_selezionate_ver)}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            # Se lo storico è già stato caricato per l'analisi e la data di inizio verifica è successiva
            # o uguale alla data di inizio dell'analisi, potremmo riutilizzarlo.
            # Altrimenti, ricarica lo storico completo.
            storico_da_usare = None
            if self.storico_caricato_per_verifica and self.date_inizio_entry_analisi.get_date() and data_inizio_ver >= self.date_inizio_entry_analisi.get_date():
                 # Verifica se il range copre anche la fine necessaria per il lookahead
                 # Questo diventa complesso, per ora ricarichiamo se la data inizio verifica è diversa dalla data inizio analisi
                 # o se non c'è storico caricato
                 if self.date_inizio_entry_analisi.get_date() == data_inizio_ver and (not self.date_fine_entry_analisi.get_date() or self.date_fine_entry_analisi.get_date() >= data_inizio_ver + timedelta(days=colpi_ver*3)): # Stima approssimativa
                     self._log_to_gui("Utilizzo storico precedentemente caricato per la verifica.")
                     storico_da_usare = self.storico_caricato_per_verifica
            
            if not storico_da_usare:
                self._log_to_gui("Ricaricamento storico completo per verifica...")
                storico_da_usare = carica_storico_completo(cartella_dati, app_logger=self._log_to_gui)


            if not storico_da_usare:
                messagebox.showinfo("Risultato Verifica", "Nessun dato storico per la verifica.")
                self._log_to_gui("Nessun dato storico per la verifica.")
                return

            verifica_giocata_manuale(
                numeri_da_verificare, ruote_gioco_selezionate_ver,
                data_inizio_ver, colpi_ver, storico_da_usare,
                app_logger=self._log_to_gui # Passa il logger
            )
            messagebox.showinfo("Verifica Completata", "Verifica terminata. Vedi risultati.")
        except Exception as e:
            messagebox.showerror("Errore Verifica", f"Errore: {e}"); self._log_to_gui(f"ERRORE VERIFICA: {e}")
            import traceback; self._log_to_gui(traceback.format_exc())
        finally: self.master.config(cursor="")


# --- BLOCCO PRINCIPALE DI ESECUZIONE ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LottoAnalyzerApp(root)
    root.mainloop()