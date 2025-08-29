import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import scipy.stats as stats
import time

@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float  # taux sans risque
    sigma: float  # volatilité
    initial_rate: float
    spread_credit: float
    recovery_rate: float
    kappa: float = 0.1
    theta: float = 0.03

@dataclass 
class SwapParameters:
    """Paramètres du swap"""
    notional: float
    maturity: float
    fixed_rate: float
    payment_frequency: int
    is_payer: bool
    swap_id: str = "Swap1"  # Identifiant pour le netting

@dataclass
class CollateralParameters:
    """Paramètres du collatéral (CSA - Credit Support Annex)"""
    threshold: float  # Seuil en dessous duquel pas de collatéral
    minimum_transfer: float  # Montant minimum de transfert (MTA)
    margin_period: int  # Période de risque de marge en jours (MPR)
    haircut: float  # Décote sur le collatéral
    frequency: int  # Fréquence d'appel de marge (jours)
    initial_margin: float = 0.0  # Marge initiale

def generate_correlated_variables_for_wwr(Nmc: int, rate_paths: np.ndarray, 
                                        correlation_wwr: float) -> np.ndarray:
    """Version améliorée pour le WWR"""
    rate_mean = np.mean(rate_paths, axis=1)
    rate_final = rate_paths[:, -1]
    rate_systemic = 0.7 * rate_mean + 0.3 * rate_final
    
    rate_normalized = (rate_systemic - np.mean(rate_systemic)) / np.std(rate_systemic)
    
    Z_independent = np.random.normal(0, 1, Nmc)
    Z_correlated = (correlation_wwr * rate_normalized + 
                   np.sqrt(1 - correlation_wwr**2) * Z_independent)
    
    return Z_correlated

class VasicekModel:
    """Modèle de Vasicek avec paramètres alignés au rapport"""
    
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """Simule les trajectoires de taux"""
        n_steps = int(round(T / dt))
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0
        
        dW = np.random.normal(0, np.sqrt(dt), (Nmc, n_steps))
        
        for i in range(n_steps):
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * dW[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
            rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.05)
            
        return rate_paths
    
    def zero_coupon_bond(self, tau: float, r_current: np.ndarray) -> np.ndarray:
        """Prix du zéro-coupon Vasicek"""
        if tau <= 0:
            return np.ones_like(r_current)
            
        if self.kappa == 0:
            B = tau
            A = np.exp(-self.theta * tau + (self.sigma**2 * tau**3) / 6)
        else:
            B = (1 - np.exp(-self.kappa * tau)) / self.kappa
            term1 = (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - tau)
            term2 = (self.sigma**2 * B**2) / (4 * self.kappa)
            A = np.exp(term1 - term2)
        
        return A * np.exp(-B * r_current)

class InterestRateSwap:
    """Swap de taux avec valorisation complète"""
    
    def __init__(self, params: SwapParameters, market_data: MarketData, 
                 rate_model: VasicekModel):
        self.params = params
        self.market_data = market_data
        self.rate_model = rate_model
        self.payment_dates = self._generate_payment_dates()
        
        if self.params.fixed_rate == 0.0:
            self.params.fixed_rate = self._calculate_atm_rate()
        
    def _generate_payment_dates(self) -> np.ndarray:
        dt = 1.0 / self.params.payment_frequency
        return np.arange(dt, self.params.maturity + dt, dt)
    
    def _calculate_atm_rate(self) -> float:
        """Calcul du taux swap ATM"""
        dt = 1.0 / self.params.payment_frequency
        
        zcb_prices = []
        for date in self.payment_dates:
            price = self.rate_model.zero_coupon_bond(
                date, np.array([self.market_data.initial_rate])
            )[0]
            zcb_prices.append(price)
        
        zcb_prices = np.array(zcb_prices)
        annuity = np.sum(dt * zcb_prices)
        
        if annuity > 0:
            return (1 - zcb_prices[-1]) / annuity
        else:
            return self.market_data.initial_rate
    
    def calculate_npv_paths(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """Calcul de la NPV par re-valorisation complète"""
        Nmc, n_steps = rate_paths.shape
        npv_paths = np.zeros_like(rate_paths)
        dt_payment = 1.0 / self.params.payment_frequency

        for i in range(n_steps - 1):
            t = time_grid[i]
            r_t = rate_paths[:, i]
            
            future_payment_dates = self.payment_dates[self.payment_dates > t]
            
            if len(future_payment_dates) == 0:
                npv_paths[:, i] = 0.0
                continue
            
            # Jambe fixe
            pv_fixed_leg = np.zeros(Nmc)
            for payment_date in future_payment_dates:
                tau = payment_date - t
                if tau > 0:
                    zc_prices = self.rate_model.zero_coupon_bond(tau, r_t)
                    pv_fixed_leg += zc_prices * dt_payment
            
            pv_fixed_leg *= self.params.fixed_rate * self.params.notional

            # Jambe flottante
            maturity_tau = self.params.maturity - t
            if maturity_tau > 0:
                zc_maturity = self.rate_model.zero_coupon_bond(maturity_tau, r_t)
                pv_float_leg = self.params.notional * (1 - zc_maturity)
            else:
                pv_float_leg = np.zeros(Nmc)

            # NPV selon position
            if self.params.is_payer:
                npv_paths[:, i] = pv_float_leg - pv_fixed_leg
            else:
                npv_paths[:, i] = pv_fixed_leg - pv_float_leg
                
        return npv_paths

class CollateralizedExposure:
    """Gestion de l'exposition avec collatéral"""
    
    def __init__(self, collateral_params: CollateralParameters):
        self.params = collateral_params
        
    def calculate_collateral_amount(self, npv: np.ndarray) -> np.ndarray:
        """
        Calcule le montant de collatéral requis selon les règles CSA
        """
        collateral = np.zeros_like(npv)
        
        # Application du threshold
        exposure_above_threshold = np.maximum(npv - self.params.threshold, 0)
        
        # Application du MTA (Minimum Transfer Amount)
        collateral = np.where(
            exposure_above_threshold > self.params.minimum_transfer,
            exposure_above_threshold * (1 - self.params.haircut),
            0
        )
        
        return collateral
    
    def calculate_collateralized_exposure(self, npv_paths: np.ndarray, 
                                         time_grid: np.ndarray) -> np.ndarray:
        """
        Calcule l'exposition en tenant compte du collatéral
        
        L'exposition avec collatéral prend en compte:
        - Le délai de liquidation (MPR - Margin Period of Risk)
        - La fréquence des appels de marge
        - Les seuils et montants minimums
        """
        Nmc, n_steps = npv_paths.shape
        collateralized_exposure = np.zeros_like(npv_paths)
        collateral_held = np.zeros(Nmc)
        
        # Conversion du MPR en pas de temps
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1/12
        mpr_steps = max(1, int(self.params.margin_period / 365 / dt))
        margin_call_frequency = max(1, int(self.params.frequency / 365 / dt))
        
        for i in range(n_steps):
            # Mise à jour du collatéral selon la fréquence
            if i % margin_call_frequency == 0:
                collateral_held = self.calculate_collateral_amount(npv_paths[:, i])
            
            # Exposition = NPV - Collatéral détenu
            # Mais on doit considérer le MPR (période de risque)
            if i + mpr_steps < n_steps:
                # Exposition future potentielle pendant le MPR
                future_npv = npv_paths[:, i + mpr_steps]
                collateralized_exposure[:, i] = np.maximum(
                    future_npv - collateral_held, 0
                )
            else:
                # Proche de la maturité
                collateralized_exposure[:, i] = np.maximum(
                    npv_paths[:, i] - collateral_held, 0
                )
        
        return collateralized_exposure

class NettingSet:
    """Gestion d'un ensemble de swaps avec netting"""
    
    def __init__(self, swaps: List[InterestRateSwap]):
        self.swaps = swaps
        
    def calculate_netted_npv_paths(self, rate_paths: np.ndarray, 
                                   time_grid: np.ndarray) -> np.ndarray:
        """
        Calcule la NPV nette du portefeuille de swaps
        Le netting réduit l'exposition en compensant les positions
        """
        # Initialisation avec le premier swap
        netted_npv = self.swaps[0].calculate_npv_paths(rate_paths, time_grid)
        
        # Addition des autres swaps
        for swap in self.swaps[1:]:
            npv_swap = swap.calculate_npv_paths(rate_paths, time_grid)
            netted_npv += npv_swap
        
        return netted_npv

class DefaultModel:
    """Modèle de défaut avec intensité constante"""
    
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate
        
    def simulate_default_times(self, T: float, Nmc: int, 
                             Z_latent: Optional[np.ndarray] = None) -> np.ndarray:
        """Simule les temps de défaut via copule"""
        if Z_latent is None:
            Z_latent = np.random.normal(0, 1, Nmc)
        
        U = stats.norm.cdf(-Z_latent)
        U = np.clip(U, 1e-10, 1-1e-10)
        
        return -np.log(U) / self.lambda_default

class CVAEngine:
    """Moteur CVA avec support du collatéral et du netting"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_exposure_metrics(self, npv_paths: np.ndarray, time_grid: np.ndarray,
                                  label: str = "") -> dict:
        """Calcule les métriques d'exposition"""
        positive_exposure = np.maximum(npv_paths, 0)
        
        ee = np.mean(positive_exposure, axis=0)
        ene = np.mean(np.minimum(npv_paths, 0), axis=0)
        pfe_95 = np.percentile(positive_exposure, 95, axis=0)
        pfe_99 = np.percentile(positive_exposure, 99, axis=0)
        epe = np.mean(ee)
        ene_avg = np.mean(ene)
        
        return {
            'label': label,
            'ee': ee, 
            'ene': ene,
            'pfe_95': pfe_95, 
            'pfe_99': pfe_99,
            'epe': epe,
            'ene_avg': ene_avg,
            'max_pfe': np.max(pfe_95)
        }
    
    def calculate_stochastic_discount_factors(self, rate_paths: np.ndarray, 
                                            time_grid: np.ndarray) -> np.ndarray:
        """Calcul des facteurs d'actualisation stochastiques"""
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.0
        
        n_rate_steps = rate_paths.shape[1]
        n_time_steps = len(time_grid)
        rates_for_integration = rate_paths[:, :min(n_rate_steps, n_time_steps)]
        
        integrated_rates = np.cumsum(rates_for_integration * dt, axis=1)
        
        return np.exp(-integrated_rates)
    
    def calculate_cva_direct(self, npv_paths: np.ndarray, 
                           default_times: np.ndarray,
                           time_grid: np.ndarray,
                           rate_paths: np.ndarray,
                           label: str = "") -> dict:
        """CVA avec actualisation stochastique"""
        Nmc = len(default_times)
        losses = np.zeros(Nmc)
        lgd = 1 - self.market_data.recovery_rate
        
        discount_factors = self.calculate_stochastic_discount_factors(rate_paths, time_grid)
        
        for j in range(Nmc):
            default_time = default_times[j]
            
            if default_time <= time_grid[-1]:
                default_idx = np.searchsorted(time_grid, default_time, side='right') - 1
                default_idx = np.clip(default_idx, 0, len(time_grid) - 1)
                
                exposure = max(0, npv_paths[j, default_idx])
                discount = discount_factors[j, default_idx]
                losses[j] = lgd * exposure * discount
        
        cva_direct = np.mean(losses)
        cva_std_error = np.std(losses, ddof=1) / np.sqrt(Nmc)
        
        return {
            'label': label,
            'cva_direct': cva_direct,
            'std_error': cva_std_error,
            'confidence_95': 1.96 * cva_std_error,
            'losses': losses
        }

def create_swap_portfolio(market_data: MarketData, rate_model: VasicekModel) -> List[InterestRateSwap]:
    """
    Crée un portefeuille de swaps pour tester le netting
    Mélange de positions payeur et receveur avec différentes maturités
    """
    swaps = []
    
    # Swap 1: Payeur 5Y (comme l'original)
    swap1_params = SwapParameters(
        notional=1_000_000,
        maturity=5.0,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True,
        swap_id="Swap1_Payer_5Y"
    )
    swaps.append(InterestRateSwap(swap1_params, market_data, rate_model))
    
    # Swap 2: Receveur 3Y (position opposée, maturité plus courte)
    swap2_params = SwapParameters(
        notional=800_000,
        maturity=3.0,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=False,
        swap_id="Swap2_Receiver_3Y"
    )
    swaps.append(InterestRateSwap(swap2_params, market_data, rate_model))
    
    # Swap 3: Payeur 7Y (même direction, maturité plus longue)
    swap3_params = SwapParameters(
        notional=600_000,
        maturity=7.0,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True,
        swap_id="Swap3_Payer_7Y"
    )
    swaps.append(InterestRateSwap(swap3_params, market_data, rate_model))
    
    # Swap 4: Receveur 5Y (position opposée, même maturité)
    swap4_params = SwapParameters(
        notional=400_000,
        maturity=5.0,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=False,
        swap_id="Swap4_Receiver_5Y"
    )
    swaps.append(InterestRateSwap(swap4_params, market_data, rate_model))
    
    return swaps

def run_comprehensive_cva_analysis() -> Dict:
    """Analyse complète: CVA standard, avec collatéral et avec netting"""
    
    print("="*80)
    print("ANALYSE CVA COMPLETE: COLLATERAL ET NETTING")
    print("="*80)
    
    # Paramètres de simulation
    Nmc = 10000  # Réduit pour rapidité, augmenter pour plus de précision
    T = 7.0  # Maturité max pour couvrir tous les swaps
    dt = 1/12
    
    # Paramètres de marché
    market_data = MarketData(
        r=0.02,
        sigma=0.025,
        initial_rate=0.02,
        theta=0.04,
        spread_credit=0.015,
        recovery_rate=0.4,
        kappa=0.3
    )
    
    # Paramètres de collatéral (CSA typique)
    collateral_params = CollateralParameters(
        threshold=100_000,       # Seuil de 100k EUR
        minimum_transfer=10_000,  # MTA de 10k EUR
        margin_period=10,         # MPR de 10 jours (standard ISDA)
        haircut=0.02,            # Haircut de 2%
        frequency=1,             # Appel de marge quotidien
        initial_margin=0.0
    )
    
    print("\nCONFIGURATION:")
    print(f"  Simulations Monte Carlo: {Nmc:,}")
    print(f"  Horizon temporel: {T} ans")
    print(f"  Pas de temps: {dt:.4f} (mensuel)")
    
    print("\nPARAMETRES DE COLLATERAL:")
    print(f"  Threshold: {collateral_params.threshold:,.0f} EUR")
    print(f"  MTA: {collateral_params.minimum_transfer:,.0f} EUR")
    print(f"  MPR: {collateral_params.margin_period} jours")
    print(f"  Haircut: {collateral_params.haircut:.1%}")
    print(f"  Fréquence marge: {collateral_params.frequency} jour(s)")
    
    # 1. Initialisation des modèles
    print("\n1. INITIALISATION DES MODELES...")
    rate_model = VasicekModel(
        market_data.initial_rate, market_data.kappa, 
        market_data.theta, market_data.sigma
    )
    
    # 2. Création du portefeuille de swaps
    print("\n2. CREATION DU PORTEFEUILLE DE SWAPS...")
    swaps = create_swap_portfolio(market_data, rate_model)
    print(f"  Nombre de swaps: {len(swaps)}")
    for swap in swaps:
        print(f"    - {swap.params.swap_id}: Notional={swap.params.notional:,.0f}, "
              f"Taux fixe={swap.params.fixed_rate:.3%}")
    
    # 3. Simulation des trajectoires
    print("\n3. SIMULATION DES TRAJECTOIRES DE TAUX...")
    n_steps = int(round(T / dt))
    time_grid = np.linspace(0, T, n_steps + 1)
    rate_paths = rate_model.simulate_paths(T, dt, Nmc)
    
    # 4. Calcul des expositions
    print("\n4. CALCUL DES EXPOSITIONS...")
    
    # 4.1 Swap unique (référence)
    single_swap = swaps[0]
    npv_single = single_swap.calculate_npv_paths(rate_paths, time_grid)
    
    # 4.2 Portefeuille avec netting
    netting_set = NettingSet(swaps)
    npv_netted = netting_set.calculate_netted_npv_paths(rate_paths, time_grid)
    
    # 4.3 Exposition avec collatéral
    collateral_manager = CollateralizedExposure(collateral_params)
    exposure_collateralized = collateral_manager.calculate_collateralized_exposure(
        npv_single, time_grid
    )
    
    # 4.4 Netting + Collatéral
    exposure_netted_collateralized = collateral_manager.calculate_collateralized_exposure(
        npv_netted, time_grid
    )
    
    # 5. Moteur CVA
    cva_engine = CVAEngine(market_data)
    
    # Métriques d'exposition
    metrics_single = cva_engine.calculate_exposure_metrics(npv_single, time_grid, "Single Swap")
    metrics_netted = cva_engine.calculate_exposure_metrics(npv_netted, time_grid, "Netted Portfolio")
    metrics_collat = cva_engine.calculate_exposure_metrics(exposure_collateralized, time_grid, "With Collateral")
    metrics_netted_collat = cva_engine.calculate_exposure_metrics(exposure_netted_collateralized, time_grid, "Netted + Collateral")
    
    # 6. Simulation des défauts
    print("\n5. SIMULATION DES DEFAUTS...")
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    default_model = DefaultModel(lambda_cp, market_data.recovery_rate)
    
    # Sans WWR
    default_times_no_wwr = default_model.simulate_default_times(T, Nmc)
    
    # Avec WWR
    correlation_wwr = -0.5
    Z_correlated = generate_correlated_variables_for_wwr(Nmc, rate_paths, correlation_wwr)
    default_times_wwr = default_model.simulate_default_times(T, Nmc, Z_correlated)
    
    # 7. Calcul du CVA pour tous les scénarios
    print("\n6. CALCUL DU CVA POUR TOUS LES SCENARIOS...")
    
    results = {}
    
    # Sans WWR
    results['single_no_wwr'] = cva_engine.calculate_cva_direct(
        npv_single, default_times_no_wwr, time_grid, rate_paths, "Single - No WWR"
    )
    
    results['netted_no_wwr'] = cva_engine.calculate_cva_direct(
        npv_netted, default_times_no_wwr, time_grid, rate_paths, "Netted - No WWR"
    )
    
    results['collat_no_wwr'] = cva_engine.calculate_cva_direct(
        exposure_collateralized, default_times_no_wwr, time_grid, rate_paths, "Collateral - No WWR"
    )
    
    results['netted_collat_no_wwr'] = cva_engine.calculate_cva_direct(
        exposure_netted_collateralized, default_times_no_wwr, time_grid, rate_paths, "Netted+Collat - No WWR"
    )
    
    # Avec WWR
    results['single_wwr'] = cva_engine.calculate_cva_direct(
        npv_single, default_times_wwr, time_grid, rate_paths, "Single - WWR"
    )
    
    results['netted_wwr'] = cva_engine.calculate_cva_direct(
        npv_netted, default_times_wwr, time_grid, rate_paths, "Netted - WWR"
    )
    
    results['collat_wwr'] = cva_engine.calculate_cva_direct(
        exposure_collateralized, default_times_wwr, time_grid, rate_paths, "Collateral - WWR"
    )
    
    results['netted_collat_wwr'] = cva_engine.calculate_cva_direct(
        exposure_netted_collateralized, default_times_wwr, time_grid, rate_paths, "Netted+Collat - WWR"
    )
    
    # 8. Affichage des résultats
    print("\n" + "="*80)
    print("RESULTATS CVA (EUR)")
    print("="*80)
    print(f"{'Scenario':<30} {'Sans WWR':>15} {'Avec WWR':>15} {'Impact WWR':>15}")
    print("-"*75)
    
    scenarios = [
        ('Single Swap', 'single_no_wwr', 'single_wwr'),
        ('Portfolio avec Netting', 'netted_no_wwr', 'netted_wwr'),
        ('Single avec Collateral', 'collat_no_wwr', 'collat_wwr'),
        ('Netting + Collateral', 'netted_collat_no_wwr', 'netted_collat_wwr')
    ]
    
    for label, key_no_wwr, key_wwr in scenarios:
        cva_no_wwr = results[key_no_wwr]['cva_direct']
        cva_wwr = results[key_wwr]['cva_direct']
        impact = (cva_wwr / cva_no_wwr - 1) * 100 if cva_no_wwr > 0 else 0
        print(f"{label:<30} {cva_no_wwr:>15,.0f} {cva_wwr:>15,.0f} {impact:>14.1f}%")
    
    # 9. Analyse des bénéfices
    print("\n" + "="*80)
    print("ANALYSE DES BENEFICES DES MITIGANTS")
    print("="*80)
    
    base_cva = results['single_wwr']['cva_direct']
    
    print(f"\nCVA de référence (Single Swap avec WWR): {base_cva:,.0f} EUR")
    print("\nRéduction du CVA par mitigant:")
    
    # Bénéfice du netting
    netting_benefit = base_cva - results['netted_wwr']['cva_direct']
    netting_benefit_pct = (netting_benefit / base_cva) * 100
    print(f"  Netting seul:              -{netting_benefit:>10,.0f} EUR ({netting_benefit_pct:>6.1f}% réduction)")
    
    # Bénéfice du collatéral
    collat_benefit = base_cva - results['collat_wwr']['cva_direct']
    collat_benefit_pct = (collat_benefit / base_cva) * 100
    print(f"  Collatéral seul:          -{collat_benefit:>10,.0f} EUR ({collat_benefit_pct:>6.1f}% réduction)")
    
    # Bénéfice combiné
    combined_benefit = base_cva - results['netted_collat_wwr']['cva_direct']
    combined_benefit_pct = (combined_benefit / base_cva) * 100
    print(f"  Netting + Collatéral:     -{combined_benefit:>10,.0f} EUR ({combined_benefit_pct:>6.1f}% réduction)")
    
    # 10. Métriques d'exposition
    print("\n" + "="*80)
    print("METRIQUES D'EXPOSITION")
    print("="*80)
    print(f"{'Scenario':<30} {'EPE (EUR)':>15} {'Max PFE 95%':>15}")
    print("-"*60)
    
    for metrics in [metrics_single, metrics_netted, metrics_collat, metrics_netted_collat]:
        print(f"{metrics['label']:<30} {metrics['epe']:>15,.0f} {metrics['max_pfe']:>15,.0f}")
    
    # Retour des résultats pour graphiques
    return {
        'results': results,
        'metrics': {
            'single': metrics_single,
            'netted': metrics_netted,
            'collateralized': metrics_collat,
            'netted_collateralized': metrics_netted_collat
        },
        'paths': {
            'rates': rate_paths,
            'npv_single': npv_single,
            'npv_netted': npv_netted,
            'exposure_collat': exposure_collateralized,
            'exposure_netted_collat': exposure_netted_collateralized
        },
        'time_grid': time_grid,
        'swaps': swaps
    }

def plot_comprehensive_results(analysis_results: Dict):
    """Génère les graphiques comparatifs"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Comparaison des profils d'exposition
    ax1 = plt.subplot(2, 3, 1)
    time_grid = analysis_results['time_grid']
    
    metrics_to_plot = [
        ('single', 'Single Swap', 'blue'),
        ('netted', 'With Netting', 'green'),
        ('collateralized', 'With Collateral', 'orange'),
        ('netted_collateralized', 'Netting + Collateral', 'red')
    ]
    
    for key, label, color in metrics_to_plot:
        ee = analysis_results['metrics'][key]['ee']
        ax1.plot(time_grid[:len(ee)], ee, label=label, color=color, linewidth=2)
    
    ax1.set_title('Profils d\'Expected Exposure', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Temps (années)')
    ax1.set_ylabel('Expected Exposure (EUR)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparaison CVA (bar chart)
    ax2 = plt.subplot(2, 3, 2)
    
    scenarios = ['Single', 'Netted', 'Collateral', 'Net+Collat']
    cva_no_wwr = [
        analysis_results['results']['single_no_wwr']['cva_direct'],
        analysis_results['results']['netted_no_wwr']['cva_direct'],
        analysis_results['results']['collat_no_wwr']['cva_direct'],
        analysis_results['results']['netted_collat_no_wwr']['cva_direct']
    ]
    cva_wwr = [
        analysis_results['results']['single_wwr']['cva_direct'],
        analysis_results['results']['netted_wwr']['cva_direct'],
        analysis_results['results']['collat_wwr']['cva_direct'],
        analysis_results['results']['netted_collat_wwr']['cva_direct']
    ]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, cva_no_wwr, width, label='Sans WWR', color='skyblue')
    bars2 = ax2.bar(x + width/2, cva_wwr, width, label='Avec WWR', color='salmon')
    
    ax2.set_title('Comparaison CVA', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CVA (EUR)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Réduction du CVA (waterfall chart simplifié)
    ax3 = plt.subplot(2, 3, 3)
    
    base_cva = analysis_results['results']['single_wwr']['cva_direct']
    reductions = [
        base_cva,
        analysis_results['results']['netted_wwr']['cva_direct'],
        analysis_results['results']['collat_wwr']['cva_direct'],
        analysis_results['results']['netted_collat_wwr']['cva_direct']
    ]
    
    labels = ['Base\n(Single)', 'Netting', 'Collateral', 'Net+Collat']
    colors = ['darkblue', 'green', 'orange', 'red']
    
    bars = ax3.bar(labels, reductions, color=colors, alpha=0.7)
    ax3.set_title('Impact des Mitigants sur le CVA', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CVA avec WWR (EUR)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les pourcentages de réduction
    for i, (bar, val) in enumerate(zip(bars, reductions)):
        if i > 0:
            reduction_pct = (1 - val/base_cva) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., val + base_cva*0.02,
                    f'-{reduction_pct:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='darkred')
    
    # 4. Distribution des expositions
    ax4 = plt.subplot(2, 3, 4)
    
    # Prendre un point dans le temps (mi-parcours)
    mid_point = len(time_grid) // 2
    
    exposures_to_compare = [
        (analysis_results['paths']['npv_single'][:, mid_point], 'Single Swap', 'blue'),
        (analysis_results['paths']['npv_netted'][:, mid_point], 'With Netting', 'green'),
        (analysis_results['paths']['exposure_collat'][:, mid_point], 'With Collateral', 'orange')
    ]
    
    for exposure, label, color in exposures_to_compare:
        ax4.hist(exposure, bins=50, alpha=0.5, label=label, color=color, density=True)
    
    ax4.set_title(f'Distribution des Expositions (t={time_grid[mid_point]:.1f} ans)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Exposition (EUR)')
    ax4.set_ylabel('Densité')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Evolution temporelle du collatéral
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculer le collatéral moyen dans le temps
    collateral_params = CollateralParameters(
        threshold=100_000,
        minimum_transfer=10_000,
        margin_period=10,
        haircut=0.02,
        frequency=1,
        initial_margin=0.0
    )
    
    collateral_manager = CollateralizedExposure(collateral_params)
    
    mean_npv = np.mean(analysis_results['paths']['npv_single'], axis=0)
    mean_collateral = collateral_manager.calculate_collateral_amount(mean_npv)
    
    ax5.plot(time_grid, mean_npv, label='NPV moyenne', color='blue', linewidth=2)
    ax5.plot(time_grid, mean_collateral, label='Collatéral moyen', color='orange', linewidth=2)
    ax5.fill_between(time_grid, 0, mean_collateral, alpha=0.3, color='orange')
    ax5.axhline(y=collateral_params.threshold, color='red', linestyle='--', 
                label=f'Threshold ({collateral_params.threshold:,.0f})')
    
    ax5.set_title('NPV et Collatéral dans le Temps', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Temps (années)')
    ax5.set_ylabel('Montant (EUR)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Tableau récapitulatif
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Créer le tableau de synthèse
    table_data = [
        ['Métrique', 'Single', 'Netting', 'Collateral', 'Net+Collat'],
        ['EPE (k€)', 
         f"{analysis_results['metrics']['single']['epe']/1000:.1f}",
         f"{analysis_results['metrics']['netted']['epe']/1000:.1f}",
         f"{analysis_results['metrics']['collateralized']['epe']/1000:.1f}",
         f"{analysis_results['metrics']['netted_collateralized']['epe']/1000:.1f}"],
        ['CVA sans WWR (€)',
         f"{analysis_results['results']['single_no_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['netted_no_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['collat_no_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['netted_collat_no_wwr']['cva_direct']:.0f}"],
        ['CVA avec WWR (€)',
         f"{analysis_results['results']['single_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['netted_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['collat_wwr']['cva_direct']:.0f}",
         f"{analysis_results['results']['netted_collat_wwr']['cva_direct']:.0f}"],
        ['Réduction vs Base (%)',
         '0.0',
         f"{(1-analysis_results['results']['netted_wwr']['cva_direct']/analysis_results['results']['single_wwr']['cva_direct'])*100:.1f}",
         f"{(1-analysis_results['results']['collat_wwr']['cva_direct']/analysis_results['results']['single_wwr']['cva_direct'])*100:.1f}",
         f"{(1-analysis_results['results']['netted_collat_wwr']['cva_direct']/analysis_results['results']['single_wwr']['cva_direct'])*100:.1f}"]
    ]
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style du tableau
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # First column
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F5F5F5')
    
    ax6.set_title('Tableau de Synthèse', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Analyse CVA: Impact du Collatéral et du Netting', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Démarrage de l'analyse CVA avec Collatéral et Netting...")
    print("Cela peut prendre quelques minutes selon le nombre de simulations...")
    
    # Lancer l'analyse complète
    analysis_results = run_comprehensive_cva_analysis()
    
    # Générer les graphiques
    plot_comprehensive_results(analysis_results)
    
    print("\nAnalyse terminée avec succès!")
    print("\nCONCLUSIONS PRINCIPALES:")
    print("1. Le netting réduit significativement l'exposition en compensant les positions")
    print("2. Le collatéral limite l'exposition mais avec un délai (MPR)")
    print("3. La combinaison netting + collatéral offre la meilleure protection")
    print("4. Le WWR augmente le CVA dans tous les scénarios")
    print("5. Les mitigants sont essentiels pour la gestion du risque de contrepartie")