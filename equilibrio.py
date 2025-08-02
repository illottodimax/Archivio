import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import threading
import queue
from itertools import combinations
from datetime import datetime
import locale
import os
import traceback

# Assicurati di aver installato tkcalendar: pip install tkcalendar
from tkcalendar import DateEntry

# ==============================================================================
# CLASSE PER LA GESTIONE DELL'ARCHIVIO
# ==============================================================================
class ArchivioLotto:
    def __init__(self, status_queue):
        self.status_queue = status_queue; self.estrazioni_per_ruota = {}; self.dati_per_analisi = {}; self.date_ordinate = []
        self.GITHUB_USER = "illottodimax"; self.GITHUB_REPO = "Archivio"; self.GITHUB_BRANCH = "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari','CA': 'Cagliari','FI': 'Firenze','GE': 'Genova','MI': 'Milano','NA': 'Napoli','PA': 'Palermo','RO': 'Roma','TO': 'Torino','VE': 'Venezia','NZ': 'Nazionale'}
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
    def _log_status(self, message):
        tag = 'error' if "ERRORE" in message else 'info'; self.status_queue.put((message, tag))
    def inizializza(self):
        self._log_status("Inizio caricamento archivio..."); total_ruote = len(self.RUOTE_DISPONIBILI)
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            self._log_status(f"Caricamento {ruota_nome} ({i+1}/{total_ruote})...")
            try: self.estrazioni_per_ruota[ruota_key] = self._carica_singola_ruota(ruota_key)
            except Exception as e: self._log_status(f"ERRORE: Impossibile caricare {ruota_nome}: {e}"); self.estrazioni_per_ruota[ruota_key] = None
        self._prepara_dati_per_analisi()
    def _carica_singola_ruota(self, ruota_key):
        response = requests.get(self.URL_RUOTE[ruota_key], timeout=15); response.raise_for_status()
        return self._parse_estrazioni(response.text.strip().split('\n'))
    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split()
            if len(parts) >= 7:
                try:
                    data = datetime.strptime(parts[0], '%Y/%m/%d').date(); numeri = [int(n) for n in parts[2:7] if n.isdigit()]
                    if len(numeri) == 5: parsed_data.append({'data': data, 'numeri': numeri})
                except (ValueError, IndexError): pass
        return parsed_data
    def _prepara_dati_per_analisi(self):
        self._log_status("Allineamento e indicizzazione dati...")
        tutte_le_date = sorted({e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni if estrazioni})
        self.date_ordinate = tutte_le_date; self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_DISPONIBILI} for data in self.date_ordinate}
        for ruota_key, estrazioni in self.estrazioni_per_ruota.items():
            if estrazioni:
                for estrazione in estrazioni:
                    if estrazione['data'] in self.dati_per_analisi: self.dati_per_analisi[estrazione['data']][ruota_key] = estrazione['numeri']
        self._log_status(f"Indicizzate {len(self.date_ordinate)} date di estrazione.")

# ==============================================================================
# CLASSE PRINCIPALE DELL'APPLICAZIONE GUI
# ==============================================================================
class DetectorUniversaleApp:
    def __init__(self, master):
        self.master = master; master.title("Detector di Equilibri (v.5.3)"); master.geometry("950x750"); master.minsize(950, 600)
        self.output_queue = queue.Queue(); self.archivio = ArchivioLotto(self.output_queue); self.nome_a_chiave = {v: k for k, v in self.archivio.RUOTE_DISPONIBILI.items()}
        style = ttk.Style(master); style.theme_use('vista'); style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold')); style.configure('TButton', font=('Segoe UI', 10, 'bold')); style.configure('Status.TLabel', font=('Segoe UI', 10, 'bold'), foreground='#003366')
        self.modalita_values = ["Coppie (Mono Ruota) - Attivi", "Coppie (Mono Ruota) - Rotti", "Coppie (Multi Ruota) - Attivi", "Coppie (Multi Ruota) - Rotti", "Numero (Multi Ruota) - Attivo", "Numero (Multi Ruota) - Rotto"]
        self.modalita_var = tk.StringVar(value=self.modalita_values[0]); self.ruota1_var = tk.StringVar(value="Bari"); self.ruota2_var = tk.StringVar(value="Cagliari"); self.ciclo_op_var = tk.IntVar(value=9)
        self._crea_widgets(); self._processa_coda_output(); self.modalita_var.trace_add("write", self.on_modalita_change)

    def _crea_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10"); main_frame.pack(expand=True, fill="both"); main_frame.rowconfigure(2, weight=1); main_frame.columnconfigure(0, weight=1)
        param_frame = ttk.LabelFrame(main_frame, text="Parametri di Analisi", padding=(10, 5)); param_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10)); param_frame.columnconfigure(1, weight=1); param_frame.columnconfigure(3, weight=1)
        ttk.Label(param_frame, text="Modalità:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="e")
        modalita_combo = ttk.Combobox(param_frame, textvariable=self.modalita_var, values=self.modalita_values, state="readonly"); modalita_combo.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        self.ruota1_label = ttk.Label(param_frame, text="Ruota:"); self.ruota1_label.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="e")
        self.ruota1_combo = ttk.Combobox(param_frame, textvariable=self.ruota1_var, values=list(self.archivio.RUOTE_DISPONIBILI.values()), state="readonly"); self.ruota1_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.ruota2_label = ttk.Label(param_frame, text="Ruota 2:"); self.ruota2_label.grid(row=1, column=2, padx=(10, 5), pady=5, sticky="e")
        self.ruota2_combo = ttk.Combobox(param_frame, textvariable=self.ruota2_var, values=list(self.archivio.RUOTE_DISPONIBILI.values()), state="readonly"); self.ruota2_combo.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        ttk.Label(param_frame, text="Lunghezza Ciclo:").grid(row=2, column=0, padx=(0, 5), pady=5, sticky="e")
        ttk.Spinbox(param_frame, from_=5, to=50, textvariable=self.ciclo_op_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="Data Inizio:").grid(row=3, column=0, padx=(0, 5), pady=5, sticky="e")
        self.data_inizio_entry = DateEntry(param_frame, date_pattern='dd/mm/yyyy', width=12, background='darkblue', foreground='white', borderwidth=2, locale='it_IT'); self.data_inizio_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w"); self.data_inizio_entry.set_date(datetime(2005, 1, 1))
        ttk.Label(param_frame, text="Data Fine:").grid(row=3, column=2, padx=(10, 5), pady=5, sticky="e")
        self.data_fine_entry = DateEntry(param_frame, date_pattern='dd/mm/yyyy', width=12, background='darkblue', foreground='white', borderwidth=2, locale='it_IT'); self.data_fine_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        self.colpi_mancanti_var = tk.StringVar(value="Colpi alla fine del ciclo: -"); ttk.Label(param_frame, textvariable=self.colpi_mancanti_var, style='Status.TLabel').grid(row=4, column=0, columnspan=4, pady=(10, 0))
        self.analyze_button = ttk.Button(main_frame, text="AVVIA ANALISI", command=self.start_analisi); self.analyze_button.grid(row=1, column=0, ipady=10, sticky="ew", pady=(10, 0))
        result_frame = ttk.LabelFrame(main_frame, text="Risultati Analisi", padding=10); result_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0)); result_frame.rowconfigure(0, weight=1); result_frame.columnconfigure(0, weight=1)
        self.output_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("Courier New", 9)); self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.tag_config('header', font=('Courier New', 10, 'bold'), foreground='#005a9e'); self.output_text.tag_config('result', font=('Courier New', 9), foreground='#C62828'); self.output_text.tag_config('info', foreground='#00695C'); self.output_text.tag_config('error', foreground='red', font=('Courier New', 9, 'bold'))
        self.on_modalita_change()

    def on_modalita_change(self, *args):
        modalita = self.modalita_var.get(); self.colpi_mancanti_var.set("Colpi alla fine del ciclo: -")
        is_mono_ruota = "Mono Ruota" in modalita
        self.ruota2_label.grid_forget() if is_mono_ruota else self.ruota2_label.grid(row=1, column=2, padx=(10, 5), pady=5, sticky="e")
        self.ruota2_combo.grid_forget() if is_mono_ruota else self.ruota2_combo.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        self.ruota1_label.config(text="Ruota:" if is_mono_ruota else "Ruota 1:")

    def _log(self, message, tag='info'): self.output_queue.put((message, tag))
    def _processa_coda_output(self):
        try:
            while True:
                message, tag = self.output_queue.get_nowait()
                if message == "ANALISI_FINITA": self.analyze_button.config(state="normal"); continue
                if message == "clear": self.output_text.config(state="normal"); self.output_text.delete("1.0", tk.END); self.output_text.config(state="disabled"); continue
                self.output_text.config(state="normal"); self.output_text.insert(tk.END, message + "\n", tag); self.output_text.config(state="disabled"); self.output_text.see(tk.END)
        except queue.Empty: pass
        self.master.after(100, self._processa_coda_output)

    def start_analisi(self):
        modalita = self.modalita_var.get()
        if "Mono Ruota" not in modalita and self.ruota1_var.get() == self.ruota2_var.get(): messagebox.showerror("Errore", "Seleziona due ruote differenti."); return
        if self.data_inizio_entry.get_date() > self.data_fine_entry.get_date(): messagebox.showerror("Errore", "La data di inizio non può essere dopo la data di fine."); return
        self.analyze_button.config(state="disabled"); self._log("clear")
        threading.Thread(target=self.run_analisi_worker, daemon=True).start()

    def run_analisi_worker(self):
        try:
            if not self.archivio.dati_per_analisi: self.archivio.inizializza()
            modalita = self.modalita_var.get()
            is_coppie = "Coppie" in modalita
            is_attivi = "Attivi" in modalita or "Attivo" in modalita
            is_multi = "Multi Ruota" in modalita
            self.run_analysis(is_coppie, is_attivi, is_multi)
        except Exception as e:
            self._log(f"ERRORE: Impossibile completare l'analisi. Causa: {e}\n{traceback.format_exc()}", 'error')
        finally:
            self._log("ANALISI_FINITA")

    def _formatta_riga(self, headers, *args):
        widths = {'COPPIA': 10, 'NUMERO': 10, 'CICLI EQ.': 11, 'RIT. ATT.': 11, 'MAX CICLI DISEQ.': 18, 'CICLI PRECEDENTI': 18, 'ROTTURA (colpi fa)': 20}
        row_parts = []
        for i, header in enumerate(headers):
            val_str = str(args[i]) if args else header
            if i == 0 and args:
                soggetto = args[0]
                val_str = f"{soggetto[0]}-{soggetto[1]}" if isinstance(soggetto, tuple) else str(soggetto)
            align = '<' if i == 0 else '^'
            row_parts.append(f"{val_str:{align}{widths[header]}}")
        return ' | '.join(row_parts)

    def conta_numero_in_ciclo(self, n, c): return sum(1 for _, nums in c if n in nums)

    def _get_data(self, multi_ruota):
        ciclo_op = self.ciclo_op_var.get(); data_inizio = self.data_inizio_entry.get_date(); data_fine = self.data_fine_entry.get_date()
        date_nel_range = [d for d in self.archivio.date_ordinate if data_inizio <= d <= data_fine]
        r1_key = self.nome_a_chiave[self.ruota1_var.get()]
        r2_key = self.nome_a_chiave[self.ruota2_var.get()] if multi_ruota else None
        estrazioni = []
        if multi_ruota:
            for data in date_nel_range:
                n1, n2 = self.archivio.dati_per_analisi[data][r1_key], self.archivio.dati_per_analisi[data][r2_key]
                if n1 and n2: estrazioni.append({'data': data, 'r1': n1, 'r2': n2})
        else:
            for data in date_nel_range:
                n1 = self.archivio.dati_per_analisi[data][r1_key]
                if n1: estrazioni.append({'data': data, 'r1': n1})
        if len(estrazioni) < ciclo_op: self._log(f"ERRORE: Trovate solo {len(estrazioni)} estrazioni valide. Servono almeno {ciclo_op}.", 'error'); self.colpi_mancanti_var.set("Colpi alla fine del ciclo: N/A"); return None
        colpi_nel_ciclo = len(estrazioni) % ciclo_op
        colpi_mancanti = ciclo_op - colpi_nel_ciclo if colpi_nel_ciclo != 0 else 0
        self.colpi_mancanti_var.set(f"Colpi alla fine del ciclo: {colpi_mancanti}")
        num_utili = (len(estrazioni) // ciclo_op) * ciclo_op
        if num_utili == 0: self._log(f"Nessun ciclo completo trovato nel range ({len(estrazioni)} estrazioni).", "error"); return None
        cicli = [estrazioni[-num_utili:][i:i + ciclo_op] for i in range(0, num_utili, ciclo_op)]
        self._log(f"Analisi su {len(estrazioni)} estrazioni, formati {len(cicli)} cicli da {ciclo_op} colpi.", 'info')
        return estrazioni, cicli

    def run_analysis(self, is_coppie, is_attivi, is_multi):
        data = self._get_data(is_multi)
        if not data: return
        estrazioni, cicli = data
        risultati = []
        soggetti = combinations(range(1, 91), 2) if is_coppie else range(1, 91)

        for soggetto in soggetti:
            def check_equilibrio(ciclo):
                c_r1 = [(c['data'], c['r1']) for c in ciclo]
                if is_coppie:
                    n1, n2 = soggetto
                    eq = self.conta_numero_in_ciclo(n1, c_r1) == self.conta_numero_in_ciclo(n2, c_r1)
                    if is_multi:
                        c_r2 = [(c['data'], c['r2']) for c in ciclo]
                        eq = eq and self.conta_numero_in_ciclo(n1, c_r2) == self.conta_numero_in_ciclo(n2, c_r2)
                else: # is_numero
                    c_r2 = [(c['data'], c['r2']) for c in ciclo]
                    eq = self.conta_numero_in_ciclo(soggetto, c_r1) == self.conta_numero_in_ciclo(soggetto, c_r2)
                return eq

            eq_attuale = check_equilibrio(cicli[-1])
            if (is_attivi and not eq_attuale) or (not is_attivi and eq_attuale): continue

            start_index = len(cicli) - 2 if not is_attivi else len(cicli) -1
            eq_consecutivi = 0
            if start_index >= 0:
                for i in range(start_index, -1, -1):
                    if check_equilibrio(cicli[i]): eq_consecutivi += 1
                    else: break
            
            if is_attivi and eq_attuale: eq_consecutivi += 1

            if eq_consecutivi > 0:
                ritardo = 0
                check_nums = soggetto if isinstance(soggetto, tuple) else (soggetto,)
                for e in reversed(estrazioni):
                    numeri_estrazione = e['r1'] + (e.get('r2', [])) if is_multi else e['r1']
                    if any(n in numeri_estrazione for n in check_nums): break
                    ritardo += 1
                risultati.append({'soggetto': soggetto, 'eq': eq_consecutivi, 'rit': ritardo})

        risultati.sort(key=lambda x: (x['eq'], x['rit']), reverse=True)
        
        soggetto_header = "COPPIA" if is_coppie else "NUMERO"
        if is_attivi: headers = [soggetto_header, 'CICLI EQ.', 'RIT. ATT.', 'MAX CICLI DISEQ.']
        else: headers = [soggetto_header, 'CICLI PRECEDENTI', 'ROTTURA (colpi fa)', 'MAX CICLI DISEQ.']
        
        self._log(self._formatta_riga(headers), 'header')
        for res in risultati[:30]:
            # Calcolo max_diseq
            max_diseq = 0; d_attuale = 0
            for ciclo in cicli:
                if not check_equilibrio(ciclo): d_attuale +=1
                else: max_diseq = max(max_diseq, d_attuale); d_attuale = 0
            max_diseq = max(max_diseq, d_attuale)
            self._log(self._formatta_riga(headers, res['soggetto'], res['eq'], res['rit'], max_diseq), 'result')

if __name__ == "__main__":
    try: locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
    except locale.Error:
        try: locale.setlocale(locale.LC_TIME, 'Italian_Italy.1252')
        except locale.Error: locale.setlocale(locale.LC_TIME, '')
    root = tk.Tk()
    app = DetectorUniversaleApp(root)
    root.mainloop()