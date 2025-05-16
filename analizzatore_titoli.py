
import pandas as pd
import tkinter as tk
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tkinter import ttk
from tkinter import scrolledtext, Listbox, Button, Frame, messagebox, END, Entry, Label
from datetime import datetime, timedelta
import time
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path

# ---- MODULI GUI ----
import tkinter as tk
from tkinter import scrolledtext, Listbox, Button, Frame, messagebox, END, Entry, Label
# from PIL import Image, ImageTk # Pillow per immagini
import subprocess
import sys
import threading

# ---- CONFIGURAZIONE LOGGING ----
# Configura il logging UNA SOLA VOLTA all'inizio.
logging.basicConfig(
    level=logging.INFO, # Mantenuto DEBUG per il troubleshooting
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout # Output di default alla console
)

# ---- CONFIGURAZIONE APPLICAZIONE ----
CHARTS_DIRECTORY = "stock_charts_output"

@dataclass
class RSIConfig:
    period: int = 14
    oversold: int = 30
    overbought: int = 70

@dataclass
class MovingAveragesConfig:
    short_term: int = 50
    long_term: int = 200

@dataclass
class ScreeningCriteriaConfig:
    min_market_cap: float = 5e9
    max_pe: float = 35.0
    min_dividend_yield: float = 0.5
    min_roe: float = 12.0
    debt_to_equity_max: float = 1.5

@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'QCOM', 'INTC', 'AMD', 'ASML', 'TXN', 'MU',
        'CRM', 'ADBE', 'ORCL', 'SAP', 'NOW', 'SNOW', 'MSI', 'JPM', 'V', 'MA', 'BAC', 'GS', 'BLK', 'AXP', 'WFC', 'C', 'SCHW',
        'JNJ', 'UNH', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'DHR', 'MDT', 'ISRG', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'COST', 
        'PG', 'KO', 'PEP', 'WMT', 'UL', 'CL', 'EL', 'PM', 'CAT', 'BA', 'HON', 'GE', 'RTX', 'LMT', 'UPS', 'FDX', 'DE',
        'XOM', 'CVX', 'SHEL', 'TTE', 'COP', 'LIN', 'APD', 'SHW', 'NEM', 'NEE', 'DUK', 'SO', 'AEP',
        'VZ', 'TMUS', 'CMCSA', 'T', 'DIS', 'NFLX', 'PYPL', 'SQ',
    ])
    lookback_period: int = 365
    moving_averages: MovingAveragesConfig = field(default_factory=MovingAveragesConfig)
    rsi: RSIConfig = field(default_factory=RSIConfig)
    screening_criteria: ScreeningCriteriaConfig = field(default_factory=ScreeningCriteriaConfig)

    # ---- NUOVI CAMPI PER LO SCORING TECNICO DELLE AZIONI ----
    base_technical_score: float = 50.0 # Punteggio di partenza, reso float
    technical_signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'golden_cross': 15.0,
        'death_cross': -15.0,
        'rsi_oversold': 12.0,
        'rsi_overbought': -12.0,
        'price_above_sma_short': 3.0,
        'price_below_sma_short': -2.0,
        'price_above_sma_long': 8.0,
        'price_below_sma_long': -5.0,
        'macd_bullish_cross': 12.0,
        'macd_bearish_cross': -12.0,
        'macd_above_zero': 5.0,
        'macd_below_zero': -3.0,
        'price_touch_bb_lower': 7.0, # Toccato/rotto BB inferiore
        'price_touch_bb_upper': -7.0 # Toccato/rotto BB superiore
    })

# ---- FUNZIONI DI UTILITY CORE (Logica di Analisi) ----

def fetch_historical_data(symbol: str, days: int) -> pd.DataFrame:
    logging.info(f"--- INIZIO FETCH per {symbol} ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Usiamo auto_adjust=True sperando che semplifichi, ma pronti a gestire il suo output
        df = yf.download(symbol, 
                         start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'), 
                         progress=False, 
                         auto_adjust=True, # auto_adjust=True usa 'Close' come prezzo aggiustato
                         timeout=15)

        if df.empty:
            logging.warning(f"yf.download ha restituito DataFrame vuoto per {symbol}.")
            return _generate_mock_data(symbol, days)

        logging.debug(f"[RAW yf.download] Colonne per {symbol}: {df.columns.tolist()}, Indice: {df.index.name}")

        # PASSO 1: Gestione del MultiIndex (se yfinance lo restituisce)
        # Questo è cruciale se auto_adjust=True a volte restituisce MultiIndex come ('Close', '')
        if isinstance(df.columns, pd.MultiIndex):
            logging.debug(f"Rilevato MultiIndex per {symbol}. Appiattimento prendendo il primo elemento di ogni tupla di colonna...")
            # Se le colonne sono tuple tipo ('Close', ''), ('High', ''), prendiamo il primo elemento
            df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in df.columns]
            logging.debug(f"Colonne dopo appiattimento MultiIndex per {symbol}: {df.columns.tolist()}")

        # PASSO 2: Resetta l'indice per trasformare 'Date' (o come si chiama) in una colonna
        df.reset_index(inplace=True)
        logging.debug(f"Colonne dopo reset_index per {symbol}: {df.columns.tolist()}")

        # PASSO 3: Rinomina e standardizza le colonne in lowercase
        # Mappa dai nomi attesi DOPO l'appiattimento del MultiIndex e reset_index
        # (che dovrebbero essere stringhe come 'Date', 'Open', 'Close') ai tuoi nomi standard.
        
        # Prima, mettiamo tutti i nomi di colonna attuali in lowercase per un confronto case-insensitive
        current_cols_lower = {str(col).lower().strip(): str(col) for col in df.columns}
        
        rename_map_targets = {
            'date': ['date', 'index', 'level_0'], # Possibili nomi per la colonna data dopo reset_index
            'open': ['open'],
            'high': ['high'],
            'low': ['low'],
            'adj_close': ['close'], # Con auto_adjust=True, 'Close' è il prezzo aggiustato
            'volume': ['volume']
        }
        
        df_renamed_cols = {}
        processed_col_names_original_case = []

        for target_name, source_name_options_lower in rename_map_targets.items():
            found_source = None
            for source_opt_lower in source_name_options_lower:
                if source_opt_lower in current_cols_lower:
                    found_source = current_cols_lower[source_opt_lower] # Prendi il nome originale case-sensitive
                    break
            
            if found_source:
                df_renamed_cols[target_name] = df[found_source]
                processed_col_names_original_case.append(found_source)
            else:
                if target_name in ['date', 'adj_close']: # Colonne essenziali
                    logging.error(f"Colonna essenziale per '{target_name}' (cercando tra {source_name_options_lower}) NON TROVATA per {symbol}. Colonne attuali: {df.columns.tolist()}. Fallback.")
                    return _generate_mock_data(symbol, days)
                else: # Per open, high, low, volume, se mancano, potremmo procedere con NaN o mock
                    logging.warning(f"Colonna opzionale per '{target_name}' (cercando tra {source_name_options_lower}) non trovata per {symbol}. Verrà riempita con NaN se possibile.")
                    df_renamed_cols[target_name] = pd.Series(np.nan, index=df.index)


        # Ricostruisci il DataFrame con le colonne rinominate e nell'ordine desiderato
        final_df = pd.DataFrame(df_renamed_cols)
        
        # Aggiungi eventuali colonne non mappate (raro se yfinance si comporta come previsto)
        for col_original in df.columns:
            col_original_str = str(col_original)
            if col_original_str not in processed_col_names_original_case and col_original_str.lower() not in final_df.columns:
                # Se una colonna non è stata processata e non ha un conflitto di nome (lowercase)
                # con quelle già presenti, aggiungila.
                final_df[col_original_str.lower().strip().replace(" ", "_")] = df[col_original]

        df = final_df # Sovrascrivi il df con quello processato
        logging.debug(f"Colonne finali dopo rinomina e standardizzazione per {symbol}: {df.columns.tolist()}")

        # PASSO 4: Normalizza la colonna 'date'
        try:
            df['date'] = pd.to_datetime(df['date'], errors='raise').dt.normalize()
        except Exception as date_conv_err:
            logging.error(f"Errore conversione colonna 'date' a datetime per {symbol}: {date_conv_err}. Fallback.")
            return _generate_mock_data(symbol, days)

        # PASSO 5: Assicurati che 'adj_close' (derivata da 'Close' con auto_adjust=True) sia valida
        if df['adj_close'].isnull().all():
            logging.error(f"'adj_close' è tutta NaN per {symbol}. Dati storici incompleti. Fallback.")
            return _generate_mock_data(symbol, days)
            
        # Se ti serve anche una colonna 'close' identica ad 'adj_close' (per coerenza con codice che usa 'close')
        if 'close' not in df.columns and 'adj_close' in df.columns:
            df['close'] = df['adj_close']
            logging.debug(f"Copiata 'adj_close' in 'close' per {symbol}.")

        # Verifica finale che le colonne OHLCV essenziali esistano
        essential_ohlcv = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        missing_essentials = [col for col in essential_ohlcv if col not in df.columns or df[col].isnull().all()]
        if missing_essentials:
            logging.error(f"Colonne OHLCV essenziali mancanti o tutte NaN ({missing_essentials}) per {symbol} dopo tutte le elaborazioni. Fallback. Colonne: {df.columns.tolist()}")
            return _generate_mock_data(symbol, days)

        logging.info(f"--- FINE FETCH per {symbol} --- Dati processati con successo.")
        return df

    except Exception as e:
        logging.error(f"Errore critico ESTERNO nel recupero/processing dati storici per {symbol}: {e}. Genero dati simulati.", exc_info=True)
        return _generate_mock_data(symbol, days)

def fetch_etf_info(symbol: str) -> Dict:
    logging.info(f"Recupero informazioni specifiche per ETF {symbol}...")
    etf_data = {
        'symbol': symbol,
        'name': 'N/A',
        'fund_family': 'N/A',
        'category': 'N/A',
        'total_assets': np.nan,
        'nav_price': np.nan,
        'previous_close': np.nan,
        'expense_ratio': np.nan,
        'beta': np.nan, # Beta può essere interessante anche per ETF
        'yield': np.nan, # Rendimento da dividendi dell'ETF (se presente)
        'error': None
    }
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        etf_data['name'] = info.get('longName', info.get('shortName', 'N/A'))
        etf_data['fund_family'] = info.get('fundFamily', 'N/A')
        etf_data['category'] = info.get('category', 'N/A')
        etf_data['total_assets'] = info.get('totalAssets', np.nan)
        
        # NAV può essere sotto diverse chiavi, o si usa previousClose
        etf_data['nav_price'] = info.get('navPrice', info.get('netAssets', np.nan)) # A volte netAssets è il NAV per azione
        etf_data['previous_close'] = info.get('previousClose', info.get('regularMarketPreviousClose', np.nan))
        if pd.isna(etf_data['nav_price']) and pd.notna(etf_data['previous_close']):
            logging.debug(f"NAV non trovato per ETF {symbol}, uso previous_close come proxy.")
            etf_data['nav_price'] = etf_data['previous_close'] # Fallback ragionevole

        # Expense ratio può avere nomi diversi
        exp_ratio = info.get('annualReportExpenseRatio', info.get('expenseRatio'))
        if exp_ratio is not None:
            etf_data['expense_ratio'] = exp_ratio * 100 # Converti in percentuale

        etf_data['beta'] = info.get('beta3Year', info.get('beta', np.nan)) # Preferisci beta a 3 anni se disponibile
        
        yld = info.get('yield') # Questo è il rendimento da dividendi del fondo
        if yld is not None:
            etf_data['yield'] = yld * 100 # Converti in percentuale

        logging.info(f"Informazioni ETF per {symbol} recuperate.")
        return etf_data

    except Exception as e:
        logging.error(f"Errore nel recupero informazioni ETF per {symbol}: {e}", exc_info=False) # exc_info=False per meno verbosità
        etf_data['error'] = str(e)
        return etf_data

def analyze_single_etf(symbol: str, config: Config) -> Dict:
    logging.info(f"Analisi ETF {symbol} in corso...")
    
    df_historical = fetch_historical_data(symbol, config.lookback_period)
    etf_info = fetch_etf_info(symbol) # etf_info contiene già una chiave 'error'

    # Struttura di ritorno di default in caso di errore precoce
    default_error_return = {
        'symbol': symbol, 'asset_type': 'ETF', 'info': etf_info,
        'historical_data': df_historical if isinstance(df_historical, pd.DataFrame) else pd.DataFrame(), # Assicura che sia un DF
        'current_indicators': {}, 'signals': {}, 'metrics': {}, 'evaluation': 'Errore Dati',
        'error': etf_info.get('error', "Errore non specificato nell'analisi ETF.")
    }

    if df_historical.empty or 'adj_close' not in df_historical.columns:
        msg = etf_info.get('error', '') + " Dati storici mancanti o corrotti per ETF."
        logging.error(msg.strip())
        default_error_return['error'] = msg.strip()
        return default_error_return
    
    min_rows_needed = max(config.moving_averages.long_term, config.rsi.period, 26, 20, 2)
    if len(df_historical) < min_rows_needed:
        msg = etf_info.get('error', '') + f" Non abbastanza dati storici ({len(df_historical)} righe) per indicatori ETF."
        logging.warning(msg.strip())
        default_error_return['error'] = msg.strip()
        default_error_return['historical_data'] = df_historical # Restituisci i dati parziali se li abbiamo
        return default_error_return

    if 'adj_close' not in df_historical.columns or not pd.api.types.is_numeric_dtype(df_historical['adj_close']):
        msg = etf_info.get('error', '') + " Colonna 'adj_close' invalida per analisi tecnica ETF."
        logging.error(msg.strip())
        default_error_return['error'] = msg.strip()
        return default_error_return

    # ---- Calcolo Indicatori Tecnici per ETF ----
    df_historical['sma_short'] = calculate_sma(df_historical, config.moving_averages.short_term)
    df_historical['sma_long'] = calculate_sma(df_historical, config.moving_averages.long_term)
    df_historical['rsi'] = calculate_rsi(df_historical, config.rsi.period)
    
    macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
    df_historical.ta.macd(close='adj_close', fast=macd_params['fast'], slow=macd_params['slow'], signal=macd_params['signal'], append=True)
    macd_line_col = f'MACD_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
    macd_signal_col = f'MACDs_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
    macd_hist_col = f'MACDh_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
        
    bb_params = {'length': 20, 'std': 2.0}
    df_historical.ta.bbands(close='adj_close', length=bb_params['length'], std=bb_params['std'], append=True)
    bb_lower_col = f'BBL_{bb_params["length"]}_{bb_params["std"]}'
    bb_middle_col = f'BBM_{bb_params["length"]}_{bb_params["std"]}'
    bb_upper_col = f'BBU_{bb_params["length"]}_{bb_params["std"]}'
    # --- Fine Calcolo Indicatori ETF ---

    latest_data = df_historical.iloc[-1]
    latest_price = latest_data.get('adj_close', np.nan)

    current_etf_indicators = {
        'price': latest_price,
        'sma_short': latest_data.get('sma_short', np.nan),
        'sma_long': latest_data.get('sma_long', np.nan),
        'rsi': latest_data.get('rsi', np.nan),
        'macd_line': latest_data.get(macd_line_col, np.nan),
        'macd_signal': latest_data.get(macd_signal_col, np.nan),
        'macd_hist': latest_data.get(macd_hist_col, np.nan),
        'bb_lower': latest_data.get(bb_lower_col, np.nan),
        'bb_middle': latest_data.get(bb_middle_col, np.nan),
        'bb_upper': latest_data.get(bb_upper_col, np.nan),
    }
    
    etf_signals = {
        'price_vs_sma_short': 'N/A', 'price_vs_sma_long': 'N/A', 'rsi_signal': 'Neutrale',
        'macd_status': 'N/A', 'bb_status': 'N/A'
    }

    if pd.notna(latest_price):
        if pd.notna(current_etf_indicators['sma_short']): etf_signals['price_vs_sma_short'] = "Sopra" if latest_price > current_etf_indicators['sma_short'] else "Sotto"
        if pd.notna(current_etf_indicators['sma_long']): etf_signals['price_vs_sma_long'] = "Sopra" if latest_price > current_etf_indicators['sma_long'] else "Sotto"
        if pd.notna(current_etf_indicators['rsi']):
            if current_etf_indicators['rsi'] < config.rsi.oversold: etf_signals['rsi_signal'] = "Ipervenduto"
            elif current_etf_indicators['rsi'] > config.rsi.overbought: etf_signals['rsi_signal'] = "Ipercomprato"
        if pd.notna(current_etf_indicators['macd_line']) and pd.notna(current_etf_indicators['macd_signal']):
            etf_signals['macd_status'] = "MACD > Segnale (Pot. Rialzista)" if current_etf_indicators['macd_line'] > current_etf_indicators['macd_signal'] else "MACD < Segnale (Pot. Ribassista)"
        if pd.notna(current_etf_indicators['bb_lower']) and latest_price <= current_etf_indicators['bb_lower']: etf_signals['bb_status'] = "Prezzo <= BB Inf."
        elif pd.notna(current_etf_indicators['bb_upper']) and latest_price >= current_etf_indicators['bb_upper']: etf_signals['bb_status'] = "Prezzo >= BB Sup."
        else: etf_signals['bb_status'] = "Prezzo tra Bande"

    etf_metrics = {}
    if pd.notna(latest_price) and pd.notna(etf_info.get('nav_price')) and etf_info.get('nav_price', 0) != 0:
        etf_metrics['premium_discount_nav'] = ((latest_price / etf_info['nav_price']) - 1) * 100
    else:
        etf_metrics['premium_discount_nav'] = np.nan

    etf_evaluation_points = []
    if pd.notna(etf_info.get('expense_ratio')):
        if etf_info['expense_ratio'] < 0.20: etf_evaluation_points.append("Basso ER")
        elif etf_info['expense_ratio'] > 0.75: etf_evaluation_points.append("Alto ER")
    if pd.notna(etf_metrics.get('premium_discount_nav')):
        if abs(etf_metrics['premium_discount_nav']) < 0.5: etf_evaluation_points.append("Vicino NAV")
        elif etf_metrics['premium_discount_nav'] > 2.0: etf_evaluation_points.append("Premio Sign. NAV")
        elif etf_metrics['premium_discount_nav'] < -2.0: etf_evaluation_points.append("Sconto Sign. NAV")
    if etf_signals['rsi_signal'] != 'Neutrale': etf_evaluation_points.append(f"RSI {etf_signals['rsi_signal']}")
    if "Rialzista" in etf_signals.get('macd_status',''): etf_evaluation_points.append("MACD Positivo")
    elif "Ribassista" in etf_signals.get('macd_status',''): etf_evaluation_points.append("MACD Negativo")
    
    etf_evaluation = "; ".join(etf_evaluation_points) if etf_evaluation_points else "Valutazione Neutrale/Dati Limitati"

    return {
        'symbol': symbol, 'asset_type': 'ETF', 'info': etf_info,
        'historical_data': df_historical,
        'current_indicators': current_etf_indicators, # Per coerenza con 'Azioni'
        'signals': etf_signals, 
        'metrics': etf_metrics,
        'evaluation': etf_evaluation, 
        'error': etf_info.get('error') # Propaga l'errore da fetch_etf_info
    }

def analyze_all_etfs(config: Config) -> List[Dict]:
    results = []
    logging.info(f"Avvio analisi per {len(config.symbols)} ETF...")
    for symbol in config.symbols:
        # Qui i simboli dovrebbero essere di ETF.
        # Per ora, la lista config.symbols è generica. L'utente dovrà inserire simboli ETF.
        analysis_result = analyze_single_etf(symbol, config)
        results.append(analysis_result)
        time.sleep(random.uniform(0.2, 0.5)) # Pausa leggermente ridotta per ETF
    # Potresti voler ordinare i risultati ETF per qualche metrica, es. expense_ratio (se disponibile)
    return results

def fetch_bond_info(bond_symbol: str) -> Dict:
    logging.info(f"Recupero informazioni per Obbligazione {bond_symbol}...")
    bond_data = {
        'symbol': bond_symbol,
        'name': 'N/A',
        'issuer_name': 'N/A', # Spesso non disponibile
        'coupon_rate': np.nan,
        'maturity_date': None, # Sarà un oggetto datetime se trovato
        'face_value': 1000, # Assunzione comune, ma può variare. Difficile da ottenere da yfinance.
        'last_price': np.nan, # Prezzo di mercato
        'yield_reported': np.nan, # Yield come riportato da yfinance (tipo incerto)
        'bond_rating': 'N/A', # Non disponibile da yfinance
        'currency': 'N/A',
        'error': None
    }
    try:
        ticker = yf.Ticker(bond_symbol)
        info = ticker.info

        bond_data['name'] = info.get('longName', info.get('shortName', 'N/A'))
        bond_data['currency'] = info.get('currency', 'N/A')
        
        # Per le obbligazioni, yfinance è molto incostante.
        # Alcuni campi potrebbero essere in 'summaryDetail' o altrove.
        summary = info.get('summaryDetail', {})
        
        bond_data['last_price'] = summary.get('previousClose', summary.get('regularMarketPreviousClose', info.get('regularMarketPrice', np.nan)))
        
        # Coupon e maturity sono difficili. A volte in 'typeDisp' o 'quoteType'.
        # Questo è molto euristico e probabilmente necessita di test con simboli reali.
        if 'couponRate' in info: # A volte è direttamente qui
            bond_data['coupon_rate'] = info.get('couponRate', np.nan) * 100 # Assumendo sia una frazione
        elif 'annualReportExpenseRatio' in info and info.get('quoteType') == 'BOND': # A volte yfinance mette il coupon qui per errore!
             bond_data['coupon_rate'] = info.get('annualReportExpenseRatio', np.nan) * 100


        if 'maturityDate' in info and info['maturityDate'] is not None:
            try:
                bond_data['maturity_date'] = datetime.fromtimestamp(info['maturityDate'])
            except Exception:
                logging.warning(f"Formato maturityDate non riconosciuto per {bond_symbol}: {info['maturityDate']}")
        elif 'expireDate' in info and info['expireDate'] is not None: # Altro nome possibile
             try:
                bond_data['maturity_date'] = datetime.fromtimestamp(info['expireDate'])
             except Exception:
                logging.warning(f"Formato expireDate non riconosciuto per {bond_symbol}: {info['expireDate']}")


        # Lo yield riportato da YF (se c'è)
        yld_val = summary.get('yield', info.get('yield', info.get('trailingAnnualDividendYield', np.nan))) # Molto speculativo
        if yld_val is not None and isinstance(yld_val, (int, float)):
             bond_data['yield_reported'] = yld_val * 100

        # Face value è quasi impossibile da ottenere da yfinance per obbligazioni individuali, si assume 100 o 1000.
        # Prezzi sono spesso quotati per 100 di nominale.

        logging.info(f"Informazioni Obbligazione per {bond_symbol} parzialmente recuperate.")
        return bond_data

    except Exception as e:
        logging.error(f"Errore nel recupero informazioni Obbligazione per {bond_symbol}: {e}", exc_info=False)
        bond_data['error'] = str(e)
        return bond_data

def calculate_bond_metrics(bond_info: Dict) -> Dict:
    metrics = {'current_yield': np.nan}
    try:
        coupon = bond_info.get('coupon_rate') # È già in %
        price = bond_info.get('last_price')
        face_value = bond_info.get('face_value', 100) # Assumiamo 100 se il prezzo è quotato per 100 di nominale

        if pd.notna(coupon) and pd.notna(price) and price != 0:
            # Current Yield = (Tasso Cedola Annuale * Valore Nominale) / Prezzo di Mercato
            # Se il prezzo è già in % del nominale (es. 101.5 per un prezzo di $1015 su $1000 nominale),
            # e il coupon_rate è in %, allora Current Yield = coupon_rate / (price/100)
            # Se coupon_rate è il tasso (es. 5 per 5%), e price è 101.5 (per 100 di nominale)
            # Annual coupon payment per 100 di nominale = coupon_rate
            # Current Yield = coupon_rate / price * 100 (se price è per 100 di nominale)
            # O più semplicemente: (coupon_rate / 100 * face_value_effettivo) / (price / 100 * face_value_effettivo) * 100
            # che si semplifica a (coupon_rate / price) * 100 se il prezzo è quotato per 100.
            
            # Assumendo che il prezzo sia quotato per 100 di valore nominale (comune)
            # e coupon_rate sia il tasso percentuale (es. 5 per 5%)
            metrics['current_yield'] = (coupon / price) * 100
            
        # Calcolo YTM è complesso e richiede un risolutore iterativo, omesso per ora.
        # Duration/Convexity richiedono ancora più dati.
    except Exception as e:
        logging.error(f"Errore nel calcolo metriche obbligazione per {bond_info.get('symbol')}: {e}")
    return metrics

def analyze_single_bond(bond_symbol: str, config: Config) -> Dict:
    logging.info(f"Analisi Obbligazione {bond_symbol} in corso...")
    
    bond_info = fetch_bond_info(bond_symbol)
    bond_metrics = {}

    # Per le obbligazioni, i dati storici di prezzo sono meno standard,
    # potremmo provare a prenderli ma l'analisi tecnica è meno comune/utile.
    # Per ora, ci concentriamo sulle info base.
    df_historical_bond = fetch_historical_data(bond_symbol, config.lookback_period) # Tentativo
    
    if bond_info.get('error'):
        return {
            'symbol': bond_symbol,
            'asset_type': 'Obbligazione',
            'info': bond_info,
            'metrics': bond_metrics, # Vuoto
            'historical_data': df_historical_bond, # Potrebbe essere mock
            'error': bond_info.get('error')
        }

    bond_metrics = calculate_bond_metrics(bond_info)

    return {
        'symbol': bond_symbol,
        'asset_type': 'Obbligazione',
        'info': bond_info,
        'metrics': bond_metrics,
        'historical_data': df_historical_bond, # Per eventuale grafico di prezzo
        'error': None
    }

def analyze_all_bonds(config: Config) -> List[Dict]:
    results = []
    logging.info(f"Avvio analisi per {len(config.symbols)} Obbligazioni...")
    for symbol in config.symbols:
        # L'utente dovrà inserire simboli di obbligazioni validi per Yahoo Finance
        analysis_result = analyze_single_bond(symbol, config)
        results.append(analysis_result)
        time.sleep(random.uniform(0.3, 0.7)) # Pausa
    # Potresti ordinare per yield o maturity date
    return results




def _generate_mock_data(symbol: str, days: int) -> pd.DataFrame:
    logging.info(f"Generazione dati simulati per {symbol}...")
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.to_datetime([(end_date - timedelta(days=i)) for i in range(days, -1, -1)]).normalize()
    base_price = 100 + random.random() * 200
    prices_open, prices_high, prices_low, prices_close, volumes_list = [], [], [], [], []
    for _ in range(len(dates)):
        daily_change_factor = 1 + (random.uniform(-0.025, 0.025))
        prev_close = base_price; base_price *= daily_change_factor
        open_price = prev_close * random.uniform(0.99, 1.01)
        high_price = max(open_price, base_price) * random.uniform(1.0, 1.02)
        low_price = min(open_price, base_price) * random.uniform(0.98, 1.0)
        close_price = base_price
        volume = int(random.uniform(500000, 20000000))
        prices_open.append(open_price); prices_high.append(high_price); prices_low.append(low_price); prices_close.append(close_price); volumes_list.append(volume)
    df = pd.DataFrame({'date': dates, 'open': prices_open, 'high': prices_high, 'low': prices_low, 'close': prices_close, 'adj_close': prices_close, 'volume': volumes_list})
    return df

def fetch_fundamentals(symbol: str) -> Dict:
    logging.info(f"Recupero dati fondamentali per {symbol}...")
    try:
        stock_ticker = yf.Ticker(symbol)
        info = stock_ticker.info
        dividend_yield_val = info.get('dividendYield')
        roe_val = info.get('returnOnEquity')
        fundamentals = {
            'market_cap': info.get('marketCap', np.nan), 'pe': info.get('trailingPE', np.nan), 'forward_pe': info.get('forwardPE', np.nan),
            'dividend_yield': dividend_yield_val * 100 if dividend_yield_val is not None else np.nan,
            'roe': roe_val * 100 if roe_val is not None else np.nan,
            'debt_to_equity': info.get('debtToEquity', np.nan), 'beta': info.get('beta', np.nan),
            'sector': info.get('sector', 'N/A'), 'industry': info.get('industry', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', np.nan))
        }
        if pd.isna(fundamentals['pe']): fundamentals['pe'] = fundamentals['forward_pe']
        if pd.notna(fundamentals['debt_to_equity']) and abs(fundamentals['debt_to_equity']) > 10: # Heuristic per D/E in formato %
            fundamentals['debt_to_equity'] /= 100
        return fundamentals
    except Exception as e:
        logging.error(f"Errore nel recupero dei fondamentali per {symbol}: {e}")
        return {'market_cap': np.nan, 'pe': np.nan, 'forward_pe': np.nan, 'dividend_yield': np.nan, 'roe': np.nan,
                'debt_to_equity': np.nan, 'beta': np.nan, 'sector': 'N/A - Errore', 'industry': 'N/A - Errore', 'current_price': np.nan}

def calculate_sma(df: pd.DataFrame, period: int, price_col: str = 'adj_close') -> pd.Series:
    if price_col not in df.columns or df[price_col].isnull().all():
        logging.warning(f"Colonna '{price_col}' mancante o tutta NaN per SMA in {df.columns.tolist() if isinstance(df, pd.DataFrame) else 'Non-DataFrame'}. Restituisco NaN.")
        return pd.Series(index=df.index if isinstance(df, pd.DataFrame) else None, data=np.nan, name=f'sma_{period}')
    return df[price_col].rolling(window=period, min_periods=max(1, period // 2)).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'adj_close') -> pd.Series:
    if price_col not in df.columns or df[price_col].isnull().all():
        logging.warning(f"Colonna '{price_col}' mancante o tutta NaN per RSI in {df.columns.tolist() if isinstance(df, pd.DataFrame) else 'Non-DataFrame'}. Restituisco NaN.")
        return pd.Series(index=df.index if isinstance(df, pd.DataFrame) else None, data=np.nan, name=f'rsi_{period}')
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10) # Evita divisione per zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def analyze_stock(symbol: str, config: Config) -> Dict:
    logging.info(f"Analisi Azione {symbol} in corso...")
    try:
        df_historical = fetch_historical_data(symbol, config.lookback_period)
        
        if df_historical.empty or not all(col in df_historical.columns for col in ['date', 'adj_close', 'close']):
             logging.error(f"Dati storici insufficienti o colonne chiave mancanti per {symbol} dopo fetch.")
             return {'symbol': symbol, 'asset_type': 'Azione', 'error': "Dati storici insufficienti o corrotti."}
        
        min_rows_needed = max(config.moving_averages.long_term, config.rsi.period, 26, 20, 2) # 26 per MACD, 20 per BB, 2 per prev_data
        if len(df_historical) < min_rows_needed:
            logging.warning(f"Non abbastanza righe di dati ({len(df_historical)}) per {symbol} per calcolare tutti gli indicatori (necessarie: {min_rows_needed}).")
            return {'symbol': symbol, 'asset_type': 'Azione', 'error': f"Non abbastanza dati ({len(df_historical)} righe) per l'analisi."}

        fundamentals = fetch_fundamentals(symbol)

        # ---- Calcolo Indicatori Tecnici ----
        if 'adj_close' not in df_historical.columns or not pd.api.types.is_numeric_dtype(df_historical['adj_close']):
            logging.error(f"Colonna 'adj_close' mancante o non numerica per {symbol}.")
            return {'symbol': symbol, 'asset_type': 'Azione', 'error': "Colonna 'adj_close' invalida per analisi tecnica."}

        df_historical[f'sma_short'] = calculate_sma(df_historical, config.moving_averages.short_term)
        df_historical[f'sma_long'] = calculate_sma(df_historical, config.moving_averages.long_term)
        df_historical[f'rsi'] = calculate_rsi(df_historical, config.rsi.period)
        
        # MACD
        macd_params = {'fast': 12, 'slow': 26, 'signal': 9} # Parametri standard
        df_historical.ta.macd(close='adj_close', fast=macd_params['fast'], slow=macd_params['slow'], signal=macd_params['signal'], append=True)
        macd_line_col = f'MACD_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
        macd_signal_col = f'MACDs_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
        macd_hist_col = f'MACDh_{macd_params["fast"]}_{macd_params["slow"]}_{macd_params["signal"]}'
        
        # Bande di Bollinger
        bb_params = {'length': 20, 'std': 2.0} # Parametri standard
        df_historical.ta.bbands(close='adj_close', length=bb_params['length'], std=bb_params['std'], append=True)
        bb_lower_col = f'BBL_{bb_params["length"]}_{bb_params["std"]}'
        bb_middle_col = f'BBM_{bb_params["length"]}_{bb_params["std"]}'
        bb_upper_col = f'BBU_{bb_params["length"]}_{bb_params["std"]}'
        # --- Fine Calcolo Indicatori ---

        latest_data = df_historical.iloc[-1]
        prev_data = df_historical.iloc[-2] 
        
        latest_price = latest_data.get('adj_close', np.nan)
        if pd.isna(latest_price): latest_price = latest_data.get('close', np.nan)
        if pd.isna(latest_price): latest_price = fundamentals.get('current_price', np.nan)
        if pd.isna(latest_price):
            logging.error(f"Prezzo più recente non disponibile per {symbol}.")
            return {'symbol': symbol, 'asset_type': 'Azione', 'error': "Prezzo più recente non disponibile."}

        signals = {
            'golden_cross': (pd.notna(latest_data.get('sma_short')) and pd.notna(latest_data.get('sma_long')) and pd.notna(prev_data.get('sma_short')) and pd.notna(prev_data.get('sma_long')) and latest_data.get('sma_short', np.nan) > latest_data.get('sma_long', np.nan) and prev_data.get('sma_short', np.nan) <= prev_data.get('sma_long', np.nan)),
            'death_cross': (pd.notna(latest_data.get('sma_short')) and pd.notna(latest_data.get('sma_long')) and pd.notna(prev_data.get('sma_short')) and pd.notna(prev_data.get('sma_long')) and latest_data.get('sma_short', np.nan) < latest_data.get('sma_long', np.nan) and prev_data.get('sma_short', np.nan) >= prev_data.get('sma_long', np.nan)),
            'rsi_oversold': pd.notna(latest_data.get('rsi')) and latest_data.get('rsi', 50) < config.rsi.oversold,
            'rsi_overbought': pd.notna(latest_data.get('rsi')) and latest_data.get('rsi', 50) > config.rsi.overbought,
            'price_above_sma_short': pd.notna(latest_data.get('sma_short')) and latest_price > latest_data.get('sma_short', float('-inf')),
            'price_below_sma_short': pd.notna(latest_data.get('sma_short')) and latest_price < latest_data.get('sma_short', float('inf')),
            'price_above_sma_long': pd.notna(latest_data.get('sma_long')) and latest_price > latest_data.get('sma_long', float('-inf')),
            'price_below_sma_long': pd.notna(latest_data.get('sma_long')) and latest_price < latest_data.get('sma_long', float('inf')),
            'macd_bullish_cross': (pd.notna(latest_data.get(macd_line_col)) and pd.notna(latest_data.get(macd_signal_col)) and pd.notna(prev_data.get(macd_line_col)) and pd.notna(prev_data.get(macd_signal_col)) and latest_data.get(macd_line_col, np.nan) > latest_data.get(macd_signal_col, np.nan) and prev_data.get(macd_line_col, np.nan) <= prev_data.get(macd_signal_col, np.nan)),
            'macd_bearish_cross': (pd.notna(latest_data.get(macd_line_col)) and pd.notna(latest_data.get(macd_signal_col)) and pd.notna(prev_data.get(macd_line_col)) and pd.notna(prev_data.get(macd_signal_col)) and latest_data.get(macd_line_col, np.nan) < latest_data.get(macd_signal_col, np.nan) and prev_data.get(macd_line_col, np.nan) >= prev_data.get(macd_signal_col, np.nan)),
            'macd_above_zero': pd.notna(latest_data.get(macd_line_col)) and latest_data.get(macd_line_col, np.nan) > 0,
            'macd_below_zero': pd.notna(latest_data.get(macd_line_col)) and latest_data.get(macd_line_col, np.nan) < 0,
            'price_touch_bb_lower': pd.notna(latest_data.get(bb_lower_col)) and latest_price <= latest_data.get(bb_lower_col, float('inf')),
            'price_touch_bb_upper': pd.notna(latest_data.get(bb_upper_col)) and latest_price >= latest_data.get(bb_upper_col, float('-inf')),
        }

        current_indicators_values = {
            'price': latest_price,
            'sma_short': latest_data.get('sma_short', np.nan),
            'sma_long': latest_data.get('sma_long', np.nan),
            'rsi': latest_data.get('rsi', np.nan),
            'macd_line': latest_data.get(macd_line_col, np.nan),
            'macd_signal': latest_data.get(macd_signal_col, np.nan),
            'macd_hist': latest_data.get(macd_hist_col, np.nan),
            'bb_lower': latest_data.get(bb_lower_col, np.nan),
            'bb_middle': latest_data.get(bb_middle_col, np.nan),
            'bb_upper': latest_data.get(bb_upper_col, np.nan),
        }

        technical_score = float(config.base_technical_score)
        for signal_name, is_active in signals.items():
            if is_active and signal_name in config.technical_signal_weights:
                technical_score += config.technical_signal_weights[signal_name]
        technical_score = max(0.0, min(100.0, technical_score))

        # Calcolo Punteggio Fondamentale (come prima, assicurati di usare .get() per `fundamentals`)
        fundamental_score = 0.0; achieved_score = 0.0; max_possible_score_for_available_data = 0.0
        criteria_weights = {'market_cap': 15, 'pe': 25, 'dividend_yield': 20, 'roe': 25, 'debt_to_equity': 15} # Puoi metterli in Config se vuoi
        
        if pd.notna(fundamentals.get('market_cap')):
            max_possible_score_for_available_data += criteria_weights['market_cap']
            if fundamentals['market_cap'] > config.screening_criteria.min_market_cap: achieved_score += criteria_weights['market_cap']
        pe_val = fundamentals.get('pe')
        if pd.notna(pe_val) and pe_val > 0:
            max_possible_score_for_available_data += criteria_weights['pe']
            if pe_val < config.screening_criteria.max_pe: achieved_score += criteria_weights['pe']
        div_yield_val = fundamentals.get('dividend_yield')
        if pd.notna(div_yield_val):
            max_possible_score_for_available_data += criteria_weights['dividend_yield']
            if div_yield_val > config.screening_criteria.min_dividend_yield: achieved_score += criteria_weights['dividend_yield']
        roe_val = fundamentals.get('roe')
        if pd.notna(roe_val):
            max_possible_score_for_available_data += criteria_weights['roe']
            if roe_val > config.screening_criteria.min_roe: achieved_score += criteria_weights['roe']
        de_val = fundamentals.get('debt_to_equity')
        if pd.notna(de_val):
            max_possible_score_for_available_data += criteria_weights['debt_to_equity']
            if de_val < config.screening_criteria.debt_to_equity_max: achieved_score += criteria_weights['debt_to_equity']
        
        if max_possible_score_for_available_data > 0: fundamental_score = (achieved_score / max_possible_score_for_available_data) * 100.0
        else: fundamental_score = 50.0
        fundamental_score = max(0.0, min(100.0, fundamental_score))

        overall_score = (technical_score * 0.5) + (fundamental_score * 0.5) # Puoi rendere questi pesi configurabili
        
        if overall_score >= 75: recommendation = 'FORTE ACQUISTO'
        elif overall_score >= 60: recommendation = 'ACQUISTO'
        elif overall_score >= 40: recommendation = 'MANTIENI'
        elif overall_score >= 25: recommendation = 'VENDI'
        else: recommendation = 'FORTE VENDITA'
        
        return {
            'symbol': symbol,
            'asset_type': 'Azione', # Aggiunto per chiarezza nel dizionario result
            'latest_price': latest_price,
            'fundamentals': fundamentals,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'overall_score': overall_score,
            'signals': signals,
            'current_indicators': current_indicators_values,
            'recommendation': recommendation,
            'df_historical': df_historical
        }
    except pd.errors.EmptyDataError:
        logging.error(f"EmptyDataError per {symbol} durante analisi tecnica, probabilmente yfinance non ha dati sufficienti per indicatori con pandas_ta.")
        return {'symbol': symbol, 'asset_type': 'Azione', 'error': "Dati insufficienti per indicatori pandas_ta."}
    except Exception as e:
        logging.error(f"Errore critico generico nell'analisi di {symbol}: {e}", exc_info=True)
        return {'symbol': symbol, 'asset_type': 'Azione', 'error': str(e)}

def analyze_all_stocks(config: Config) -> List[Dict]:
    results = []
    for symbol in config.symbols:
        analysis_result = analyze_stock(symbol, config)
        results.append(analysis_result)
        time.sleep(random.uniform(0.3, 0.8)) # Leggermente ridotta pausa per test
    return sorted(results, key=lambda x: x.get('overall_score', -1), reverse=True)

def create_stock_chart(stock_analysis: Dict, config: Config) -> Optional[Path]:
    try:
        symbol = stock_analysis['symbol']
        df = stock_analysis['df_historical'] # Ora contiene colonne MACD e BB
        
        if df.empty or not all(col in df.columns for col in ['date', 'adj_close']):
            logging.warning(f"Dati insufficienti o colonne 'date'/'adj_close' mancanti per grafico di {symbol}.")
            return None

        charts_path = Path(CHARTS_DIRECTORY)
        charts_path.mkdir(parents=True, exist_ok=True)

        # Determina il numero di subplot necessari (Prezzo+BB, RSI, MACD)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1, 2]}) # Aggiustato per 3 subplot
        fig.suptitle(f"Analisi Tecnica di {symbol} - {stock_analysis.get('recommendation', 'Dati Asset')}", fontsize=14)
        
        ax_price = axes[0]
        ax_rsi = axes[1]
        ax_macd = axes[2]

        # ---- Grafico Prezzi, SMA e Bande di Bollinger ----
        ax_price.plot(df['date'], df['adj_close'], label='Prezzo Adj.', color='navy', alpha=0.9, linewidth=1.2)
        if 'sma_short' in df.columns: ax_price.plot(df['date'], df['sma_short'], label=f'SMA {config.moving_averages.short_term}', color='darkorange', linestyle='--', linewidth=0.9)
        if 'sma_long' in df.columns: ax_price.plot(df['date'], df['sma_long'], label=f'SMA {config.moving_averages.long_term}', color='crimson', linestyle='--', linewidth=0.9)
        
        # Bande di Bollinger (i nomi esatti delle colonne dipendono dai parametri usati in pandas_ta)
        bb_lower_col = next((col for col in df.columns if 'BBL' in col), None) # Trova colonna Lower Band
        bb_middle_col = next((col for col in df.columns if 'BBM' in col), None) # Middle Band
        bb_upper_col = next((col for col in df.columns if 'BBU' in col), None) # Upper Band

        if bb_lower_col and bb_upper_col and bb_middle_col:
            ax_price.plot(df['date'], df[bb_middle_col], label='BB Middle', color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_price.plot(df['date'], df[bb_upper_col], label='BB Upper', color='lightgray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_price.plot(df['date'], df[bb_lower_col], label='BB Lower', color='lightgray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_price.fill_between(df['date'], df[bb_lower_col], df[bb_upper_col], color='gainsboro', alpha=0.2, interpolate=True)
        
        ax_price.set_title(f'Prezzo, SMA e Bande di Bollinger', fontsize=9)
        ax_price.set_ylabel('Prezzo ($)', fontsize=8)
        ax_price.legend(fontsize=7, loc='upper left')
        ax_price.grid(True, linestyle=':', alpha=0.5)
        ax_price.tick_params(axis='y', labelsize=7)
        
        # ---- Grafico RSI (come prima) ----
        if 'rsi' in df.columns:
            # ... (codice grafico RSI identico a prima, usa ax_rsi) ...
            ax_rsi.plot(df['date'], df['rsi'], label=f'RSI ({config.rsi.period})', color='purple', linewidth=1.2)
            ax_rsi.axhline(config.rsi.overbought, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_rsi.axhline(config.rsi.oversold, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_rsi.fill_between(df['date'], config.rsi.overbought, df['rsi'], where=(df['rsi'] >= config.rsi.overbought), color='salmon', alpha=0.3, interpolate=True)
            ax_rsi.fill_between(df['date'], config.rsi.oversold, df['rsi'], where=(df['rsi'] <= config.rsi.oversold), color='lightgreen', alpha=0.3, interpolate=True)
        ax_rsi.set_title(f'RSI', fontsize=9)
        ax_rsi.set_ylabel('RSI', fontsize=8)
        ax_rsi.set_ylim([0, 100])
        ax_rsi.legend(fontsize=7, loc='upper left')
        ax_rsi.grid(True, linestyle=':', alpha=0.5)
        ax_rsi.tick_params(axis='y', labelsize=7)

        # ---- Grafico MACD ----
        macd_line_col = next((col for col in df.columns if 'MACD_' in col and 'MACDh' not in col and 'MACDs' not in col), None)
        macd_signal_col = next((col for col in df.columns if 'MACDs_' in col), None)
        macd_hist_col = next((col for col in df.columns if 'MACDh_' in col), None)

        if macd_line_col and macd_signal_col and macd_hist_col:
            ax_macd.plot(df['date'], df[macd_line_col], label='MACD Line', color='blue', linewidth=1)
            ax_macd.plot(df['date'], df[macd_signal_col], label='Signal Line', color='red', linestyle='--', linewidth=1)
            # Colora l'istogramma MACD
            colors = ['green' if val >= 0 else 'red' for val in df[macd_hist_col]]
            ax_macd.bar(df['date'], df[macd_hist_col], label='MACD Histogram', color=colors, width=0.7, alpha=0.5) # width dipende dalla frequenza dei dati
            ax_macd.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax_macd.set_title(f'MACD (12,26,9)', fontsize=9)
        ax_macd.set_ylabel('MACD', fontsize=8)
        ax_macd.legend(fontsize=7, loc='upper left')
        ax_macd.grid(True, linestyle=':', alpha=0.5)
        ax_macd.tick_params(axis='both', labelsize=7)
        
        # Formattazione asse X comune
        plt.xticks(rotation=30, ha='right')
        ax_macd.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%y-%m-%d'))
        ax_macd.xaxis.set_major_locator(plt.MaxNLocator(10)) # Meno ticks per leggibilità
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Aggiusta per suptitle e labels
        
        chart_filename = f"{symbol}_analysis_adv.png" # Nome file diverso
        full_chart_path = charts_path / chart_filename
        plt.savefig(full_chart_path, dpi=100)
        logging.info(f"Grafico avanzato per {symbol} salvato: {full_chart_path}")
        plt.close(fig)
        return full_chart_path
    except Exception as e:
        logging.error(f"Errore creazione grafico avanzato per {stock_analysis['symbol']}: {e}", exc_info=True)
        return None

# ---- FUNZIONI GUI ----
analysis_data_store = []
generated_charts_paths = [] 
symbol_entry_widget = None 

class TextAreaHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.is_gui_destroyed = False # Flag per controllare se la GUI è stata distrutta

    def emit(self, record):
        if self.is_gui_destroyed or not self.text_widget.winfo_exists(): # Controlla se il widget esiste ancora
            self.is_gui_destroyed = True # Imposta il flag se distrutto
            return # Non tentare di scrivere su un widget distrutto
            
        msg = self.format(record)
        try:
            current_state = self.text_widget.cget("state")
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=current_state) # Ripristina lo stato precedente (potrebbe essere 'disabled')
        except tk.TclError: # Errore comune se il widget è stato distrutto
            self.is_gui_destroyed = True


def open_image_viewer(image_path: Path):
    if not image_path.exists():
        messagebox.showerror("Errore", f"File immagine non trovato: {image_path}")
        return
    try:
        if sys.platform == "win32": os.startfile(str(image_path)) # os.startfile vuole str
        elif sys.platform == "darwin": subprocess.Popen(["open", str(image_path)])
        else: subprocess.Popen(["xdg-open", str(image_path)])
    except Exception as e:
        messagebox.showerror("Errore apertura immagine", f"Impossibile aprire {image_path}: {e}")

def run_analysis_threaded_gui(app_config: Config, text_area: scrolledtext.ScrolledText, chart_listbox: Listbox, run_button: Button):
    global analysis_data_store, generated_charts_paths, symbol_entry_widget
    
    def update_gui_before_analysis():
        run_button.config(state=tk.DISABLED, text="Analisi in corso...")
        text_area.configure(state='normal')
        text_area.delete('1.0', END)
        # text_area.configure(state='disabled') # Lascia normale per il logger
        chart_listbox.delete(0, END)
    
    text_area.after(0, update_gui_before_analysis) # Schedula aggiornamenti GUI nel thread principale
    generated_charts_paths.clear()

    raw_symbols = symbol_entry_widget.get().split(',')
    app_config.symbols = [s.strip().upper() for s in raw_symbols if s.strip()]

    if not app_config.symbols:
        logging.info("Nessun simbolo inserito. Analisi annullata.")
        text_area.after(0, lambda: run_button.config(state=tk.NORMAL, text="Avvia Analisi"))
        return

    logging.info(f"Avvio analisi per: {', '.join(app_config.symbols)}")
    
    try:
        analysis_data_store = analyze_all_stocks(app_config) # Questa è la parte lunga
        
        # Log del report finale
        logging.info("\n" + "="*50 + "\nREPORT DI INVESTIMENTO\n" + "="*50)
        for result in analysis_data_store:
            if 'error' in result:
                logging.error(f"\n{result['symbol']}: ERRORE - {result['error']}")
                continue
            
            logging.info(f"\n--- {result['symbol']} ---")
            logging.info(f"Raccomandazione: {result['recommendation']} (Punteggio: {result['overall_score']:.1f})")
            logging.info(f"  Prezzo: ${result['latest_price']:.2f}")
            f = result['fundamentals']
            logging.info("  Dati Fondamentali:")
            logging.info(f"    Cap: ${f.get('market_cap', 0)/1e9:.2f}B" if pd.notna(f.get('market_cap')) else "    Cap: N/A")
            logging.info(f"    P/E: {f.get('pe'):.2f}" if pd.notna(f.get('pe')) else "    P/E: N/A")
            logging.info(f"    DivY: {f.get('dividend_yield'):.2f}%" if pd.notna(f.get('dividend_yield')) else "    DivY: N/A")
            logging.info(f"    ROE: {f.get('roe'):.2f}%" if pd.notna(f.get('roe')) else "    ROE: N/A")
            logging.info(f"    D/E: {f.get('debt_to_equity'):.2f}" if pd.notna(f.get('debt_to_equity')) else "    D/E: N/A")
            logging.info(f"    Settore: {f.get('sector', 'N/A')}")
            logging.info(f"  Punteggio Tecnico: {result['technical_score']:.1f}/100")
            logging.info(f"  Punteggio Fondamentale: {result['fundamental_score']:.1f}/100")
            
            active_signals = [name for name, active in result['signals'].items() if active]
            if active_signals: logging.info(f"  Segnali: {', '.join(sig.replace('_', ' ').title() for sig in active_signals)}")
            
            if ('ACQUISTO' in result['recommendation'] or 'FORTE ACQUISTO' in result['recommendation']) and 'df_historical' in result:
                chart_file_path = create_stock_chart(result, app_config)
                if chart_file_path and chart_file_path.exists():
                     generated_charts_paths.append(chart_file_path)
                     # Schedula aggiornamento Listbox nel thread principale
                     chart_listbox.after(0, lambda path_name=chart_file_path.name: chart_listbox.insert(END, path_name))


        top_picks = [r for r in analysis_data_store if 'error' not in r and ('ACQUISTO' in r['recommendation'] or 'FORTE ACQUISTO' in r['recommendation'])][:3]
        logging.info("\n\n" + "="*50 + "\nMIGLIORI OPPORTUNITÀ:\n" + "="*50)
        if top_picks:
            for i, pick in enumerate(top_picks):
                logging.info(f"{i+1}. {pick['symbol']} ({pick['recommendation']}) - Punteggio: {pick['overall_score']:.1f}")
        else:
            logging.info("Nessuna opportunità di acquisto/forte acquisto identificata.")
        logging.info("\n" + "-"*50 + "\nNOTA: Report a scopo informativo.\n" + "="*50)
        logging.info("Analisi completata.")

    except Exception as e:
        logging.error(f"Errore durante l'analisi principale: {e}", exc_info=True)
    finally:
        text_area.after(0, lambda: run_button.config(state=tk.NORMAL, text="Avvia Analisi"))
        # text_area.after(0, lambda: text_area.configure(state='disabled')) # Disabilita di nuovo alla fine

def start_analysis_gui(app_config: Config, text_area: scrolledtext.ScrolledText, chart_listbox: Listbox, run_button: Button, asset_type_var: tk.StringVar): # << Aggiunto asset_type_var
    selected_asset_type = asset_type_var.get() # Ottieni il valore stringa dal Combobox
    analysis_thread = threading.Thread(target=run_master_analysis, # << Chiama una nuova funzione master
                                       args=(app_config, text_area, chart_listbox, run_button, selected_asset_type))
    analysis_thread.daemon = True
    analysis_thread.start()

# Assicurati che questa funzione sia definita DOPO
# analyze_all_stocks, analyze_all_etfs, analyze_all_bonds, e generate_report_for_gui

def run_master_analysis(app_config: Config, text_area: scrolledtext.ScrolledText, chart_listbox: Listbox, run_button: Button, asset_type: str):
    global analysis_data_store, generated_charts_paths, symbol_entry_widget
    
    # Funzione interna per aggiornamenti GUI sicuri per thread
    def schedule_gui_update(task, *args, **kwargs):
        if text_area.winfo_exists(): 
            text_area.after(0, lambda: task(*args, **kwargs))
        else:
            logging.warning("Tentativo di aggiornare la GUI dopo la sua distruzione. Operazione ignorata.")

    # Aggiornamenti iniziali della GUI
    schedule_gui_update(run_button.config, state=tk.DISABLED, text="Analisi in corso...")
    schedule_gui_update(text_area.configure, state='normal') # Lascia normale per il logger
    schedule_gui_update(text_area.delete, '1.0', END)
    schedule_gui_update(chart_listbox.delete, 0, END)
    
    generated_charts_paths.clear()
    analysis_data_store.clear()

    raw_symbols = symbol_entry_widget.get().split(',')
    app_config.symbols = [s.strip().upper() for s in raw_symbols if s.strip()]

    if not app_config.symbols:
        logging.info("Nessun simbolo inserito. Analisi annullata.")
        schedule_gui_update(run_button.config, state=tk.NORMAL, text="Avvia Analisi")
        return

    logging.info(f"Avvio analisi per {asset_type}: {', '.join(app_config.symbols)}")
    
    try:
        if asset_type == "Azioni":
            logging.info("Esecuzione analisi per Azioni...")
            analysis_data_store = analyze_all_stocks(app_config)
            generate_report_for_gui(analysis_data_store, asset_type, chart_listbox, app_config)

        elif asset_type == "ETF":
            logging.info("Esecuzione analisi per ETF...")
            analysis_data_store = analyze_all_etfs(app_config)
            generate_report_for_gui(analysis_data_store, asset_type, chart_listbox, app_config)

        elif asset_type == "Obbligazioni":
            logging.info(f"Esecuzione analisi per Obbligazioni: {', '.join(app_config.symbols)}")
            analysis_data_store = analyze_all_bonds(app_config)
            generate_report_for_gui(analysis_data_store, asset_type, chart_listbox, app_config)

        else:
            logging.error(f"Tipo di asset sconosciuto o non supportato: '{asset_type}'")
            analysis_data_store = [{'symbol': 'N/A', 'asset_type': asset_type, 'error': f'Tipo asset {asset_type} non supportato.'}]
            generate_report_for_gui(analysis_data_store, asset_type, chart_listbox, app_config)

        logging.info("Analisi completata.")

    except Exception as e:
        logging.error(f"Errore critico durante l'analisi ({asset_type}): {e}", exc_info=True)
        analysis_data_store.append({'symbol': 'ERRORE GENERALE', 'asset_type': asset_type, 'error': f'Errore imprevisto: {e}'})
        generate_report_for_gui(analysis_data_store, asset_type, chart_listbox, app_config)
    finally:
        schedule_gui_update(run_button.config, state=tk.NORMAL, text="Avvia Analisi")

# Assicurati che pandas (pd), logging, create_stock_chart, e la variabile globale generated_charts_paths
# siano definite e accessibili prima di questa funzione.
# Anche la dataclass Config deve essere definita.

def generate_report_for_gui(analysis_results: List[Dict], asset_type: str, chart_listbox: Listbox, config: Config):
    global generated_charts_paths 
    
    logging.info(f"\n" + "="*50 + f"\nREPORT DI INVESTIMENTO ({asset_type})\n" + "="*50)
    
    for result in analysis_results:
        symbol = result.get('symbol', 'N/A_SYM')
        error_message = result.get('error') # Può essere None se non ci sono errori
        
        # Stampa un errore solo se error_message è una stringa non vuota (cioè un errore effettivo)
        if error_message is not None and isinstance(error_message, str):
            logging.error(f"\n{symbol}: ERRORE - {error_message}")
            continue # Passa al prossimo risultato se c'è stato un errore significativo
        
        # Se siamo qui, o error_message è None (nessun errore grave), o non è una stringa (caso anomalo)
        logging.info(f"\n--- {symbol} ({asset_type}) ---")

        if asset_type == "Azioni":
            # Se error_message era None, l'analisi dell'azione è andata a buon fine.
            logging.info(f"Raccomandazione: {result.get('recommendation', 'N/A')} (Punteggio: {result.get('overall_score', 0.0):.1f})")
            logging.info(f"  Prezzo Attuale: ${result.get('latest_price', float('nan')):.2f}")
            
            f = result.get('fundamentals', {})
            logging.info("  Dati Fondamentali:")
            logging.info(f"    Cap: ${f.get('market_cap', 0)/1e9:.2f}B" if pd.notna(f.get('market_cap')) else "    Cap: N/A")
            logging.info(f"    P/E: {f.get('pe', float('nan')):.2f}" if pd.notna(f.get('pe')) else "    P/E: N/A")
            logging.info(f"    DivY: {f.get('dividend_yield', float('nan')):.2f}%" if pd.notna(f.get('dividend_yield')) else "    DivY: N/A")
            logging.info(f"    ROE: {f.get('roe', float('nan')):.2f}%" if pd.notna(f.get('roe')) else "    ROE: N/A")
            logging.info(f"    D/E: {f.get('debt_to_equity', float('nan')):.2f}" if pd.notna(f.get('debt_to_equity')) else "    D/E: N/A")
            logging.info(f"    Settore: {f.get('sector', 'N/A')}")
            
            logging.info(f"  Punteggio Tecnico: {result.get('technical_score', 0.0):.1f}/100")
            logging.info(f"  Punteggio Fondamentale: {result.get('fundamental_score', 0.0):.1f}/100")
            
            current_inds = result.get('current_indicators', {})
            logging.info("  Valori Indicatori Tecnici:")
            logging.info(f"    SMA {config.moving_averages.short_term}gg: {current_inds.get('sma_short', float('nan')):.2f}")
            logging.info(f"    SMA {config.moving_averages.long_term}gg: {current_inds.get('sma_long', float('nan')):.2f}")
            logging.info(f"    RSI ({config.rsi.period}): {current_inds.get('rsi', float('nan')):.1f}")
            logging.info(f"    MACD Line: {current_inds.get('macd_line', float('nan')):.2f}")
            logging.info(f"    MACD Signal: {current_inds.get('macd_signal', float('nan')):.2f}")
            logging.info(f"    MACD Hist: {current_inds.get('macd_hist', float('nan')):.2f}")
            logging.info(f"    BB Lower: {current_inds.get('bb_lower', float('nan')):.2f}")
            logging.info(f"    BB Middle: {current_inds.get('bb_middle', float('nan')):.2f}")
            logging.info(f"    BB Upper: {current_inds.get('bb_upper', float('nan')):.2f}")

            signals = result.get('signals', {})
            active_signals_desc = []
            # Costruisci la lista delle descrizioni dei segnali attivi
            if signals.get('golden_cross'): active_signals_desc.append("Golden Cross")
            if signals.get('death_cross'): active_signals_desc.append("Death Cross")
            if signals.get('rsi_oversold'): active_signals_desc.append("RSI Ipervenduto")
            if signals.get('rsi_overbought'): active_signals_desc.append("RSI Ipercomprato")
            if signals.get('macd_bullish_cross'): active_signals_desc.append("MACD Incrocio Rialzista")
            if signals.get('macd_bearish_cross'): active_signals_desc.append("MACD Incrocio Ribassista")
            if signals.get('macd_above_zero'): active_signals_desc.append("MACD Sopra Zero")
            if signals.get('macd_below_zero'): active_signals_desc.append("MACD Sotto Zero")
            if signals.get('price_touch_bb_lower'): active_signals_desc.append("Prezzo Tocca BB Inferiore")
            if signals.get('price_touch_bb_upper'): active_signals_desc.append("Prezzo Tocca BB Superiore")
            if signals.get('price_above_sma_short'): active_signals_desc.append(f"Prezzo > SMA{config.moving_averages.short_term}")
            if signals.get('price_below_sma_short'): active_signals_desc.append(f"Prezzo < SMA{config.moving_averages.short_term}")
            if signals.get('price_above_sma_long'): active_signals_desc.append(f"Prezzo > SMA{config.moving_averages.long_term}")
            if signals.get('price_below_sma_long'): active_signals_desc.append(f"Prezzo < SMA{config.moving_averages.long_term}")
            
            if active_signals_desc: 
                logging.info(f"  Segnali Tecnici Attivi: {', '.join(active_signals_desc)}")
            else:
                logging.info("  Nessun segnale tecnico specifico rilevato.")
            
            recommendation = result.get('recommendation', '')
            if ('ACQUISTO' in recommendation or 'FORTE ACQUISTO' in recommendation) and 'df_historical' in result:
                # Prepara il dizionario per create_stock_chart in modo che abbia le chiavi attese
                chart_input_data = {
                    'symbol': symbol,
                    'df_historical': result['df_historical'],
                    'recommendation': recommendation # Passa la raccomandazione effettiva
                }
                chart_file_path = create_stock_chart(chart_input_data, config)
                if chart_file_path and chart_file_path.exists():
                     generated_charts_paths.append(chart_file_path)
                     if chart_listbox.winfo_exists():
                        chart_listbox.after(0, lambda path_name=chart_file_path.name: chart_listbox.insert(END, path_name))

        elif asset_type == "ETF":
            info = result.get('info', {})
            tech_values = result.get('current_indicators', {}) 
            etf_signals = result.get('signals', {})
            metrics = result.get('metrics', {})

            logging.info(f"  Nome: {info.get('name', 'N/A')}")
            logging.info(f"  Famiglia Fondo: {info.get('fund_family', 'N/A')}")
            logging.info(f"  Categoria: {info.get('category', 'N/A')}")
            total_assets_val = info.get('total_assets')
            logging.info(f"  Patrimonio Totale (AUM): ${total_assets_val/1e6:,.0f} M" if pd.notna(total_assets_val) else "  Patrimonio Totale (AUM): N/A")
            expense_ratio_val = info.get('expense_ratio')
            logging.info(f"  Expense Ratio: {expense_ratio_val:.2f}%" if pd.notna(expense_ratio_val) else "  Expense Ratio: N/A")
            yield_val = info.get('yield')
            logging.info(f"  Rendimento (Yield): {yield_val:.2f}%" if pd.notna(yield_val) else "  Rendimento (Yield): N/A")
            beta_val = info.get('beta')
            logging.info(f"  Beta: {beta_val:.2f}" if pd.notna(beta_val) else "  Beta: N/A")
            
            price_val = tech_values.get('price')
            nav_val = info.get('nav_price')
            logging.info(f"  Prezzo Attuale: ${price_val:.2f}" if pd.notna(price_val) else "  Prezzo Attuale: N/A")
            logging.info(f"  NAV Stimato: ${nav_val:.2f}" if pd.notna(nav_val) else "  NAV Stimato: N/A")
            prem_disc_val = metrics.get('premium_discount_nav')
            logging.info(f"  Premio/Sconto vs NAV: {prem_disc_val:.2f}%" if pd.notna(prem_disc_val) else "  Premio/Sconto vs NAV: N/A")

            logging.info("  Valori Indicatori Tecnici (ETF):")
            logging.info(f"    SMA {config.moving_averages.short_term}gg: {tech_values.get('sma_short', float('nan')):.2f} ({etf_signals.get('price_vs_sma_short','N/A')})")
            logging.info(f"    SMA {config.moving_averages.long_term}gg: {tech_values.get('sma_long', float('nan')):.2f} ({etf_signals.get('price_vs_sma_long','N/A')})")
            logging.info(f"    RSI ({config.rsi.period}): {tech_values.get('rsi', float('nan')):.1f} ({etf_signals.get('rsi_signal','N/A')})")
            logging.info(f"    MACD Line: {tech_values.get('macd_line', float('nan')):.2f}")
            logging.info(f"    MACD Signal: {tech_values.get('macd_signal', float('nan')):.2f}")
            logging.info(f"    MACD Hist: {tech_values.get('macd_hist', float('nan')):.2f}")
            logging.info(f"    MACD Status: {etf_signals.get('macd_status','N/A')}")
            logging.info(f"    BB Lower: {tech_values.get('bb_lower', float('nan')):.2f}")
            logging.info(f"    BB Middle: {tech_values.get('bb_middle', float('nan')):.2f}")
            logging.info(f"    BB Upper: {tech_values.get('bb_upper', float('nan')):.2f}")
            logging.info(f"    BB Status: {etf_signals.get('bb_status','N/A')}")
            
            logging.info(f"  Valutazione ETF: {result.get('evaluation', 'N/A')}")

            if 'historical_data' in result and not result['historical_data'].empty:
                chart_data_for_etf = {'symbol': symbol, 'df_historical': result['historical_data'], 'recommendation': "Dati ETF"}
                chart_file_path = create_stock_chart(chart_data_for_etf, config)
                if chart_file_path and chart_file_path.exists():
                     generated_charts_paths.append(chart_file_path)
                     if chart_listbox.winfo_exists():
                        chart_listbox.after(0, lambda path_name=chart_file_path.name: chart_listbox.insert(END, path_name))
        
        elif asset_type == "Obbligazioni":
            info = result.get('info', {})
            metrics = result.get('metrics', {})
            logging.info(f"  Nome/Descrizione: {info.get('name', 'N/A')}")
            logging.info(f"  Valuta: {info.get('currency', 'N/A')}")
            coupon_rate = info.get('coupon_rate')
            logging.info(f"  Tasso Cedola: {coupon_rate:.2f}%" if pd.notna(coupon_rate) else "  Tasso Cedola: N/A")
            maturity_date = info.get('maturity_date')
            logging.info(f"  Data di Scadenza: {maturity_date.strftime('%Y-%m-%d')}" if maturity_date else "  Data di Scadenza: N/A")
            last_price = info.get('last_price')
            logging.info(f"  Ultimo Prezzo: {last_price:.2f}" if pd.notna(last_price) else "  Ultimo Prezzo: N/A (spesso % del nominale)")
            current_yield = metrics.get('current_yield')
            logging.info(f"  Current Yield Stimato: {current_yield:.2f}%" if pd.notna(current_yield) else "  Current Yield Stimato: N/A")
            yield_rep = info.get('yield_reported')
            logging.info(f"  Yield Riportato (YF): {yield_rep:.2f}%" if pd.notna(yield_rep) else "  Yield Riportato (YF): N/A (tipo incerto)")
            logging.info(f"  Rating Creditizio: {info.get('bond_rating', 'N/A (non da YF)')}")
            
            if 'historical_data' in result and not result['historical_data'].empty and 'adj_close' in result['historical_data'].columns:
                chart_data_for_bond = {'symbol': symbol, 'df_historical': result['historical_data'], 'recommendation': "Andamento Prezzo Obbligazione"}
                chart_file_path = create_stock_chart(chart_data_for_bond, config)
                if chart_file_path and chart_file_path.exists():
                     generated_charts_paths.append(chart_file_path)
                     if chart_listbox.winfo_exists():
                        chart_listbox.after(0, lambda path_name=chart_file_path.name: chart_listbox.insert(END, path_name))
            else:
                logging.info(f"  Nessun dato storico di prezzo sufficiente per generare grafico per l'obbligazione {symbol}.")

    # Sezione "MIGLIORI OPPORTUNITÀ"
    if asset_type == "Azioni":
        top_picks = [r for r in analysis_results if r.get('error') is None and ('ACQUISTO' in r.get('recommendation','') or 'FORTE ACQUISTO' in r.get('recommendation',''))][:3]
        logging.info("\n\n" + "="*50 + "\nMIGLIORI OPPORTUNITÀ (Azioni):\n" + "="*50)
        if top_picks:
            for i, pick in enumerate(top_picks):
                logging.info(f"{i+1}. {pick.get('symbol','N/A')} ({pick.get('recommendation','N/A')}) - Punteggio: {pick.get('overall_score',0.0):.1f}")
        else:
            logging.info("Nessuna opportunità di acquisto/forte acquisto identificata per le Azioni.")
    # Aggiungi qui una logica simile per ETF o Obbligazioni se definisci criteri di "top picks"
    # Esempio per ETF (molto basilare, basato su expense ratio basso):
    elif asset_type == "ETF":
        # Filtra ETF senza errori e con expense ratio valido
        valid_etfs_for_ranking = [
            r for r in analysis_results 
            if r.get('error') is None and 
            isinstance(r.get('info'), dict) and 
            pd.notna(r['info'].get('expense_ratio'))
        ]
        if valid_etfs_for_ranking:
            # Ordina per expense ratio (ascendente) e prendi i primi 3
            top_etfs = sorted(valid_etfs_for_ranking, key=lambda x: x['info']['expense_ratio'])[:3]
            logging.info("\n\n" + "="*50 + "\nETF CON BASSO EXPENSE RATIO (Esempio):\n" + "="*50)
            if top_etfs:
                for i, etf_pick in enumerate(top_etfs):
                    logging.info(f"{i+1}. {etf_pick.get('symbol','N/A')} - ER: {etf_pick['info'].get('expense_ratio', float('nan')):.2f}% - Nome: {etf_pick.get('info',{}).get('name','N/A')}")
            else:
                logging.info("Nessun ETF con dati su Expense Ratio trovato per il ranking.")
        else:
            logging.info("\nNessun ETF con dati sufficienti per un ranking sull'Expense Ratio.")
            
    logging.info("\n" + "-"*50 + "\nNOTA: Report a scopo informativo.\n" + "="*50)

def view_selected_chart_gui(chart_listbox: Listbox):
    selected_indices = chart_listbox.curselection()
    if not selected_indices:
        messagebox.showinfo("Info", "Nessun grafico selezionato.")
        return
    
    selected_chart_index = selected_indices[0]
    if 0 <= selected_chart_index < len(generated_charts_paths):
        chart_path_to_open = generated_charts_paths[selected_chart_index]
        open_image_viewer(chart_path_to_open)
    else:
        messagebox.showerror("Errore", "Indice grafico non valido o lista grafici non sincronizzata.")

def main_gui():
    global symbol_entry_widget

    app_config = Config()
    root = tk.Tk()
    root.title("Universal Asset Analyzer") # Titolo più generico
    root.geometry("1000x700")

    # Setup Logger per GUI (assumendo che gui_root_logger sia già configurato)
    gui_root_logger = logging.getLogger() # Prende il root logger
    # La configurazione del livello e del formato dell'handler della GUI avviene più avanti

    # ---- Frame per Controlli di Input (Tipo Asset e Simboli) ----
    controls_frame = Frame(root, pady=10)
    controls_frame.pack(fill=tk.X, padx=10)

    # Selezione Tipo Asset
    Label(controls_frame, text="Tipo Asset:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
    asset_types = ["Azioni", "ETF", "Obbligazioni"] 
    asset_type_var = tk.StringVar(value=asset_types[0]) # Valore di default
    
    asset_combobox = ttk.Combobox(controls_frame, textvariable=asset_type_var, values=asset_types, state="readonly", width=15, font=("Arial", 10))
    asset_combobox.pack(side=tk.LEFT, padx=(0, 10))
    # Potresti aggiungere una funzione da chiamare quando il tipo di asset cambia,
    # per esempio per suggerire simboli di default diversi:
    # asset_combobox.bind("<<ComboboxSelected>>", on_asset_type_change) 

    # Input Simboli
    Label(controls_frame, text="Simboli (es: AAPL,MSFT):", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0,5))
    symbol_entry_widget = Entry(controls_frame, width=50, font=("Arial", 10)) # Ridotta leggermente la larghezza per far spazio al Combobox
    symbol_entry_widget.insert(0, ",".join(app_config.symbols)) # Pre-popola con simboli di default (per Azioni)
    symbol_entry_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)


    # ---- Frame principale per output ----
    main_content_frame = Frame(root, padx=10, pady=5)
    main_content_frame.pack(fill=tk.BOTH, expand=True)

    # Output area (sinistra)
    report_outer_frame = Frame(main_content_frame, bd=1, relief=tk.SOLID)
    report_outer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
    Label(report_outer_frame, text="Output Analisi e Log:", font=("Arial", 11, "bold")).pack(anchor=tk.NW, pady=(2,2), padx=2)
    report_text_area = scrolledtext.ScrolledText(report_outer_frame, wrap=tk.WORD, font=("Courier New", 9), height=10, state='disabled')
    report_text_area.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    text_area_handler = TextAreaHandler(report_text_area)
    text_area_handler.setLevel(logging.INFO) # O logging.INFO per meno verbosità nella GUI
    text_area_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    gui_root_logger.addHandler(text_area_handler)


    # Charts area (destra)
    charts_outer_frame = Frame(main_content_frame, bd=1, relief=tk.SOLID, width=280)
    charts_outer_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
    charts_outer_frame.pack_propagate(False) 
    Label(charts_outer_frame, text="Grafici Generati:", font=("Arial", 11, "bold")).pack(anchor=tk.NW, pady=(2,2), padx=2)
    chart_listbox = Listbox(charts_outer_frame, font=("Arial", 10), selectmode=tk.SINGLE, exportselection=False)
    chart_listbox.pack(pady=5, fill=tk.BOTH, expand=True, padx=2)
    
    view_chart_button = Button(charts_outer_frame, text="Visualizza Grafico", font=("Arial", 10),
                               command=lambda: view_selected_chart_gui(chart_listbox))
    view_chart_button.pack(pady=(0,5), fill=tk.X, padx=2)

    # Pulsante di avvio
    run_button = Button(root, text="Avvia Analisi", font=("Arial", 12, "bold"), 
                        bg="#D6EAF8", activebackground="#AED6F1", relief=tk.RAISED, padx=10, pady=5)
    # Modifica la chiamata per passare asset_type_var
    run_button.config(command=lambda: start_analysis_gui(app_config, report_text_area, chart_listbox, run_button, asset_type_var))
    run_button.pack(pady=(0,10))
    
    # Funzione per gestire la chiusura della finestra
    def on_closing():
        if text_area_handler: # Assicurati che l'handler esista
            text_area_handler.is_gui_destroyed = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    logging.info("Universal Asset Analyzer avviato. Seleziona tipo asset, inserisci simboli e clicca 'Avvia Analisi'.")
    root.mainloop()

if __name__ == "__main__":
    main_gui()