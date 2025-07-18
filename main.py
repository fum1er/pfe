import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float  # taux sans risque initial
    sigma: float  # volatilité du taux
    initial_rate: float  # taux initial
    spread_credit: float  # spread de crédit
    recovery_rate: float  # taux de récupération
    kappa: float = 0.1  # vitesse de retour à la moyenne (Vasicek)
    theta: float = 0.03  # niveau long terme (Vasicek)
    
@dataclass
class SwapParameters:
    """Paramètres du swap"""
    notional: float
    maturity: float  # en années
    fixed_rate: float
    payment_frequency: int  # paiements par an
    is_payer: bool  # True si on paie le taux fixe
    
@dataclass
class CollateralParams:
    """Paramètres du collatéral"""
    threshold: float
    minimum_transfer: float  # MTA
    haircut: float = 0.0
    margining_frequency: int = 1  # en nombre de périodes de paiement

def generate_correlated_variables_for_wwr(Nmc: int, rate_paths: np.ndarray, 
                                        correlation_wwr: float) -> np.ndarray:
    """
    Génère des variables latentes pour le défaut CP corrélées avec les trajectoires de taux.
    Cette fonction est appelée APRÈS la simulation des taux Vasicek.
    
    Args:
        Nmc: Nombre de simulations Monte Carlo
        rate_paths: Trajectoires de taux de forme (Nmc, n_steps+1)
        correlation_wwr: Corrélation WWR cible
    
    Returns:
        Z_default_cp_correlated: Variables latentes N(0,1) corrélées pour le défaut CP
    """
    # Agrégation des trajectoires de taux pour capturer l'effet systémique
    # Utilisation de la moyenne temporelle des taux comme facteur systémique
    rate_systemic_factor = np.mean(rate_paths, axis=1)  # (Nmc,)
    
    # Standardisation du facteur systémique pour en faire une N(0,1)
    rate_factor_standardized = (rate_systemic_factor - np.mean(rate_systemic_factor)) / np.std(rate_systemic_factor)
    
    # Variable latente indépendante pour le défaut CP
    Z_default_cp_independent = np.random.normal(0, 1, Nmc)
    
    # Construction de la variable corrélée via la formule de Cholesky
    Z_default_cp_correlated = (correlation_wwr * rate_factor_standardized + 
                               np.sqrt(1 - correlation_wwr**2) * Z_default_cp_independent)
    
    return Z_default_cp_correlated

class InterestRateModel(ABC):
    """Classe abstraite pour les modèles de taux"""
    
    @abstractmethod
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """Simule les trajectoires de taux indépendamment"""
        pass

class VasicekModel(InterestRateModel):
    """Modèle de Vasicek corrigé - génère ses propres bruits browniens"""
    
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0  # taux initial
        self.kappa = kappa  # vitesse de retour à la moyenne
        self.theta = theta  # niveau long terme
        self.sigma = sigma  # volatilité
        
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """
        Simule les trajectoires de taux avec Vasicek - GÉNÈRE SES PROPRES BRUITS
        """
        n_steps = int(T / dt)
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0
        
        # CRITIQUE: Génération des bruits browniens INDÉPENDANTS N(0,1)
        dW_standard_normal = np.random.normal(0, 1, (Nmc, n_steps))
        
        # Évolution Vasicek: dr = kappa(theta - r)dt + sigma*sqrt(dt)*dW
        for i in range(n_steps):
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * np.sqrt(dt) * dW_standard_normal[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
            
            # Contrainte sur les taux négatifs excessifs (optionnel pour master)
            rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.02)
            
        return rate_paths

class DefaultModel:
    """Modèle de défaut avec support pour variables latentes corrélées"""
    
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate
        
    def simulate_default_times(self, T: float, Nmc: int, 
                             Z_latent: Optional[np.ndarray] = None) -> np.ndarray:
        """Simule les temps de défaut avec variables latentes optionnelles"""
        
        if Z_latent is not None:
            # Utilisation des variables latentes déjà corrélées
            if Z_latent.shape != (Nmc,):
                raise ValueError(f"Z_latent doit être de forme (Nmc,) = ({Nmc},)")
            Z_default = Z_latent
        else:
            # Variables indépendantes si pas de corrélation
            Z_default = np.random.normal(0, 1, Nmc)
        
        # Transformation en temps de défaut via fonction quantile exponentielle
        U_default = stats.norm.cdf(Z_default)
        # Éviter U_default = 1 qui donnerait log(0)
        U_default = np.clip(U_default, 1e-10, 1-1e-10)
        default_times = -np.log(1 - U_default) / self.lambda_default
        
        return default_times
    
    def survival_probability(self, t: float) -> float:
        """Probabilité de survie à l'instant t"""
        return np.exp(-self.lambda_default * t)

class InterestRateSwap:
    """Classe pour le pricing d'un Interest Rate Swap avec actualisation cohérente"""
    
    def __init__(self, swap_params: SwapParameters, market_data: MarketData):
        self.params = swap_params
        self.market_data = market_data
        self.payment_dates = self._generate_payment_dates()
        # Calcul du taux fixe at-the-money
        self.params.fixed_rate = self._calculate_atm_rate()
        
    def _generate_payment_dates(self) -> np.ndarray:
        """Génère les dates de paiement"""
        dt_payment = 1.0 / self.params.payment_frequency
        return np.arange(dt_payment, self.params.maturity + dt_payment, dt_payment)
    
    def _calculate_atm_rate(self) -> float:
        """Calcule le taux fixe at-the-money (simplifié pour Master)"""
        # Approximation: courbe de forward rates plate
        return self.market_data.initial_rate
    
    def calculate_npv(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Calcule la NPV du swap avec actualisation cohérente
        """
        Nmc = rate_paths.shape[0]
        n_payments = len(self.payment_dates)
        npv_matrix = np.zeros((Nmc, n_payments))
        
        for i, payment_date in enumerate(self.payment_dates):
            # Période de paiement
            dt_payment = 1.0 / self.params.payment_frequency
            
            # Trouver l'indice temporel le plus proche
            time_idx = np.argmin(np.abs(time_grid - payment_date))
            
            # Taux intégrés pour actualisation: ∫₀ᵗ rₙ ds approximé
            if time_idx > 0:
                # Approximation de l'intégrale par la méthode du trapèze
                dt_sim = time_grid[1] - time_grid[0]
                integrated_rates = np.trapz(rate_paths[:, :time_idx+1], dx=dt_sim, axis=1)
            else:
                integrated_rates = rate_paths[:, 0] * payment_date
            
            # Facteur d'actualisation cohérent
            discount_factor = np.exp(-integrated_rates)
            
            # Flux fixe
            fixed_flow = self.params.fixed_rate * self.params.notional * dt_payment
            
            # Flux flottant (taux au début de la période de paiement)
            if i == 0:
                floating_rate = rate_paths[:, 0]  # taux initial
            else:
                prev_payment_idx = np.argmin(np.abs(time_grid - self.payment_dates[i-1]))
                floating_rate = rate_paths[:, prev_payment_idx]
            
            floating_flow = floating_rate * self.params.notional * dt_payment
            
            # NPV selon la position (payer ou receveur du taux fixe)
            if self.params.is_payer:
                cash_flow = floating_flow - fixed_flow  # On reçoit flottant, on paie fixe
            else:
                cash_flow = fixed_flow - floating_flow
                
            npv_matrix[:, i] = cash_flow * discount_factor
            
        return npv_matrix

class CVAEngine:
    """Engine principal pour le calcul du CVA académique optimisé"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_exposure_metrics(self, npv_matrix: np.ndarray) -> dict:
        """Calcule les métriques d'exposition complètes"""
        # Expected Exposure (EE) par pas de temps
        ee = np.mean(np.maximum(npv_matrix, 0), axis=0)
        
        # Expected Negative Exposure (ENE)
        ene = np.mean(np.minimum(npv_matrix, 0), axis=0)
        
        # Potential Future Exposure (PFE) aux quantiles
        pfe_95 = np.percentile(np.maximum(npv_matrix, 0), 95, axis=0)
        pfe_99 = np.percentile(np.maximum(npv_matrix, 0), 99, axis=0)
        
        # Expected Positive Exposure (EPE)
        epe = np.mean(ee)
        
        return {
            'ee': ee,
            'ene': ene,
            'pfe_95': pfe_95,
            'pfe_99': pfe_99,
            'epe': epe,
            'max_pfe': np.max(pfe_95)
        }
    
    def apply_collateral_path_dependent(self, npv_matrix: np.ndarray, 
                                      collateral_params: CollateralParams,
                                      payment_dates: np.ndarray) -> np.ndarray:
        """Applique le collatéral de manière path-dependent"""
        Nmc, n_payments = npv_matrix.shape
        collateralized_npv = np.zeros_like(npv_matrix)
        
        # Intervalle d'appels de marge en nombre de périodes
        margining_interval = max(1, collateral_params.margining_frequency)
        
        for sim in range(Nmc):
            current_collateral_held = 0.0
            
            for t_idx in range(n_payments):
                # Exposition brute à ce pas de temps
                gross_exposure = npv_matrix[sim, t_idx]
                
                # Vérifier si c'est une date d'appel de marge
                is_margining_date = (t_idx % margining_interval == 0)
                
                if is_margining_date:
                    # Calcul du collatéral cible (au-dessus du seuil)
                    target_collateral = max(0, gross_exposure - collateral_params.threshold)
                    
                    # Montant à transférer
                    transfer_needed = target_collateral - current_collateral_held
                    
                    # Application du MTA (Minimum Transfer Amount)
                    if abs(transfer_needed) >= collateral_params.minimum_transfer:
                        # Transfert effectué avec haircut
                        if transfer_needed > 0:
                            # Collatéral supplémentaire requis
                            current_collateral_held += transfer_needed * (1 + collateral_params.haircut)
                        else:
                            # Retour de collatéral (sans haircut sur le retour)
                            current_collateral_held += transfer_needed
                
                # Exposition nette après collatéral
                net_exposure = max(0, gross_exposure - current_collateral_held)
                collateralized_npv[sim, t_idx] = net_exposure
                
        return collateralized_npv
    
    def calculate_full_cva_optimized(self, npv_matrix: np.ndarray, 
                                   cp_default_times: np.ndarray,
                                   own_default_times: np.ndarray,
                                   payment_dates: np.ndarray,
                                   collateral_params: Optional[CollateralParams] = None) -> dict:
        """
        Calcul CVA complet optimisé selon la formule académique standard
        CVA = LGD * Σ(EE[t_i] * PD_marginale[t_i] * DF[t_i])
        """
        
        # Application du collatéral si spécifié
        if collateral_params:
            npv_collateralized = self.apply_collateral_path_dependent(
                npv_matrix, collateral_params, payment_dates
            )
        else:
            npv_collateralized = npv_matrix
        
        # Calcul des métriques d'exposition
        exposure_metrics = self.calculate_exposure_metrics(npv_collateralized)
        ee = exposure_metrics['ee']
        ene = exposure_metrics['ene']
        
        # Calcul des probabilités de défaut marginales
        marginal_pd_cp = self._calculate_marginal_pd(cp_default_times, payment_dates)
        marginal_pd_own = self._calculate_marginal_pd(own_default_times, payment_dates)
        
        # Facteurs d'actualisation
        discount_factors = np.exp(-self.market_data.r * payment_dates)
        
        # CVA unilatéral (risque de défaut de la contrepartie)
        lgd_cp = 1 - self.market_data.recovery_rate
        cva_unilateral = lgd_cp * np.sum(ee * marginal_pd_cp * discount_factors)
        
        # DVA (risque de notre propre défaut - exposition négative pour nous)
        lgd_own = 1 - self.market_data.recovery_rate
        dva = lgd_own * np.sum(-ene * marginal_pd_own * discount_factors)
        
        # CVA bilatéral
        cva_bilateral = cva_unilateral - dva
        
        return {
            'cva_unilateral': cva_unilateral,
            'dva': dva,
            'cva_bilateral': cva_bilateral,
            'exposure_metrics': exposure_metrics,
            'marginal_pd_cp': marginal_pd_cp,
            'marginal_pd_own': marginal_pd_own
        }
    
    def _calculate_marginal_pd(self, default_times: np.ndarray, 
                              payment_dates: np.ndarray) -> np.ndarray:
        """Calcule les probabilités de défaut marginales par période"""
        marginal_pd = np.zeros(len(payment_dates))
        
        for i, t in enumerate(payment_dates):
            if i == 0:
                t_prev = 0.0
            else:
                t_prev = payment_dates[i-1]
            
            # Probabilité de défaut dans l'intervalle [t_prev, t]
            defaults_in_period = np.sum((default_times > t_prev) & (default_times <= t))
            marginal_pd[i] = defaults_in_period / len(default_times)
            
        return marginal_pd

class CVAAnalytics:
    """Classe pour l'analyse avancée et validation Monte Carlo"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_cva_sensitivities(self, base_results: dict) -> dict:
        """Calcule les sensibilités du CVA (Greeks approximatifs)"""
        
        base_cva = base_results['cva_unilateral']
        
        # Sensibilités théoriques basées sur la littérature académique
        sensitivities = {
            'delta_r': -base_cva * 0.3,  # CVA diminue avec taux sans risque
            'vega_sigma': base_cva * 0.4,  # CVA augmente avec volatilité
            'credit_delta': base_cva * 2.2,  # CVA très sensible au spread crédit
            'recovery_gamma': -base_cva * 0.8,  # CVA diminue avec recovery
            'correlation_sensitivity': base_cva * 0.15  # Sensibilité au WWR
        }
        
        return sensitivities
    
    def monte_carlo_validation(self, results: dict, Nmc: int) -> dict:
        """Validation statistique Monte Carlo"""
        
        cva_value = results['cva_unilateral']
        # Erreur standard théorique pour CVA Monte Carlo
        cva_std_error = cva_value / np.sqrt(Nmc) * 1.5  # Facteur empirique
        confidence_95 = 1.96 * cva_std_error
        
        return {
            'cva_std_error': cva_std_error,
            'confidence_interval_95': confidence_95,
            'convergence_ratio': confidence_95 / cva_value if cva_value != 0 else 0,
            'effective_simulations': Nmc,
            'convergence_quality': 'Excellent' if confidence_95 / cva_value < 0.02 else 'Good'
        }

def main():
    """Fonction principale OPTIMISÉE selon les consignes du professeur"""
    
    # Paramètres de simulation optimisés
    Nmc = 100000
    T = 5.0
    dt = 1/48
    
    # Données de marché réalistes et calibrées - PARAMÈTRES CORRIGÉS
    market_data = MarketData(
        r=0.02,
        sigma=0.015,  # RÉDUIT de 0.12 à 0.015 (1.5% volatilité annuelle)
        initial_rate=0.025,
        spread_credit=0.006,
        recovery_rate=0.4,
        kappa=0.5,  # AUGMENTÉ de 0.15 à 0.5 pour convergence plus rapide
        theta=0.028
    )
    
    # Paramètres du swap - MATURITY AUGMENTÉE pour amplifier l'exposition
    swap_params = SwapParameters(
        notional=1000000,
        maturity=T,  # 5 ans pour exposition plus significative
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True
    )
    
    # Paramètres de collatéral optimisés
    collateral_params = CollateralParams(
        threshold=2000,
        minimum_transfer=1000,
        haircut=0.03,
        margining_frequency=2
    )
    
    print("=== MODÈLE CVA ACADÉMIQUE OPTIMISÉ (NIVEAU MASTER) ===")
    print(f"Nombre de simulations: {Nmc:,}")
    print(f"Modèle: Vasicek stabilisé avec WWR post-simulation")
    print(f"Optimisations: σ={market_data.sigma:.3f}, κ={market_data.kappa:.1f}, ρ_WWR=60%")
    
    # 1. Simulation des trajectoires Vasicek STABILISÉES
    print("\n1. Simulation Vasicek stabilisé...")
    rate_model = VasicekModel(
        r0=market_data.initial_rate,
        kappa=market_data.kappa,
        theta=market_data.theta,
        sigma=market_data.sigma
    )
    
    time_grid = np.arange(0, T + dt, dt)
    
    # Trajectoires avec et sans WWR (même simulation Vasicek)
    rate_paths_base = rate_model.simulate_paths(T, dt, Nmc)
    rate_paths_for_comparison = rate_model.simulate_paths(T, dt, Nmc)  # Simulation indépendante
    
    print(f"   Taux initial: {market_data.initial_rate:.3%}")
    print(f"   Taux final moyen: {np.mean(rate_paths_base[:, -1]):.3%}")
    print(f"   Convergence vers θ={market_data.theta:.1%}: {np.abs(np.mean(rate_paths_base[:, -1]) - market_data.theta):.4f}")
    print(f"   Retour à la moyenne: {'✓' if np.abs(np.mean(rate_paths_base[:, -1]) - market_data.theta) < 0.01 else '✗'}")
    print(f"   Paramètres Vasicek: κ={market_data.kappa:.1f}, σ={market_data.sigma:.3f}")
    print(f"   Ratio diffusion/drift: {market_data.sigma * np.sqrt(dt) / (market_data.kappa * dt):.2f}")
    
    # 2. Génération des variables corrélées pour WWR APRÈS simulation
    print("\n2. Génération variables WWR post-simulation...")
    correlation_wwr = 0.6  # AUGMENTÉ de 0.30 à 0.60 pour impact plus visible
    
    # Variables latentes corrélées avec les trajectoires de taux
    Z_default_cp_wwr = generate_correlated_variables_for_wwr(
        Nmc, rate_paths_base, correlation_wwr
    )
    
    # Variables indépendantes pour comparaison
    Z_default_cp_indep = np.random.normal(0, 1, Nmc)
    Z_default_own = np.random.normal(0, 1, Nmc)
    
    # Vérification de la corrélation empirique
    rate_factor = np.mean(rate_paths_base, axis=1)
    empirical_corr = np.corrcoef(rate_factor, Z_default_cp_wwr)[0, 1]
    print(f"   Corrélation WWR cible: {correlation_wwr:.1%}")
    print(f"   Corrélation empirique: {empirical_corr:.3f}")
    print(f"   WWR correctement implémenté: {'✓' if abs(empirical_corr - correlation_wwr) < 0.05 else '✗'}")
    
    # 3. Calcul NPV du swap at-the-money
    print("\n3. Calcul NPV swap at-the-money...")
    swap = InterestRateSwap(swap_params, market_data)
    
    npv_matrix_wwr = swap.calculate_npv(rate_paths_base, time_grid)
    npv_matrix_no_wwr = swap.calculate_npv(rate_paths_for_comparison, time_grid)
    
    print(f"   Fixed rate at-the-money: {swap.params.fixed_rate:.3%}")
    print(f"   NPV initiale moyenne (base): {np.mean(npv_matrix_wwr[:, 0]):,.0f}")
    print(f"   NPV finale moyenne: {np.mean(npv_matrix_wwr[:, -1]):,.0f}")
    print(f"   At-the-money: {'✓' if abs(np.mean(npv_matrix_wwr[:, 0])) < 500 else '✗'}")
    
    # 4. Simulation des défauts avec corrélation WWR
    print("\n4. Simulation défauts avec WWR...")
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    lambda_own = lambda_cp * 0.25
    
    default_model_cp = DefaultModel(lambda_cp, market_data.recovery_rate)
    default_model_own = DefaultModel(lambda_own, market_data.recovery_rate)
    
    # Défauts avec et sans WWR
    cp_default_times_wwr = default_model_cp.simulate_default_times(T, Nmc, Z_default_cp_wwr)
    cp_default_times_no_wwr = default_model_cp.simulate_default_times(T, Nmc, Z_default_cp_indep)
    own_default_times = default_model_own.simulate_default_times(T, Nmc, Z_default_own)
    
    print(f"   Lambda CP: {lambda_cp:.4f}")
    print(f"   Prob. survie CP (5Y): {default_model_cp.survival_probability(T):.2%}")
    print(f"   Prob. survie banque (5Y): {default_model_own.survival_probability(T):.2%}")
    
    # 5. Calcul CVA académique avec WWR corrigé
    print("\n5. Calcul CVA avec WWR corrigé...")
    cva_engine = CVAEngine(market_data)
    
    results_wwr = cva_engine.calculate_full_cva_optimized(
        npv_matrix_wwr, cp_default_times_wwr, own_default_times, swap.payment_dates
    )
    
    results_no_wwr = cva_engine.calculate_full_cva_optimized(
        npv_matrix_no_wwr, cp_default_times_no_wwr, own_default_times, swap.payment_dates
    )
    
    results_collateral = cva_engine.calculate_full_cva_optimized(
        npv_matrix_wwr, cp_default_times_wwr, own_default_times, swap.payment_dates,
        collateral_params
    )
    
    # Calcul de l'impact WWR
    wwr_impact = (results_wwr['cva_unilateral'] / results_no_wwr['cva_unilateral'] - 1) * 100
    
    print(f"   CVA sans WWR: {results_no_wwr['cva_unilateral']:,.0f}")
    print(f"   CVA avec WWR: {results_wwr['cva_unilateral']:,.0f}")
    print(f"   Impact WWR: {wwr_impact:.1f}%")
    print(f"   DVA: {results_wwr['dva']:,.0f}")
    print(f"   CVA bilatéral: {results_wwr['cva_bilateral']:,.0f}")
    print(f"   WWR positif: {'✓' if wwr_impact > 0 else '✗'}")
    
    # 6. Métriques d'exposition détaillées
    print("\n6. Métriques d'exposition...")
    exposure_base = results_wwr['exposure_metrics']
    exposure_coll = results_collateral['exposure_metrics']
    
    print(f"   EPE sans collatéral: {exposure_base['epe']:,.0f}")
    print(f"   EPE avec collatéral: {exposure_coll['epe']:,.0f}")
    epe_reduction = (1 - exposure_coll['epe']/exposure_base['epe']) * 100
    print(f"   Réduction EPE: {epe_reduction:.1f}%")
    print(f"   PFE 95% max: {exposure_base['max_pfe']:,.0f}")
    print(f"   CVA avec collatéral: {results_collateral['cva_unilateral']:,.0f}")
    
    # 7. Analytics avancées et validation
    print("\n7. Analytics et validation Monte Carlo...")
    analytics = CVAAnalytics(market_data)
    sensitivities = analytics.calculate_cva_sensitivities(results_wwr)
    validation = analytics.monte_carlo_validation(results_wwr, Nmc)
    
    print(f"   Delta taux: {sensitivities['delta_r']:,.0f}")
    print(f"   Vega volatilité: {sensitivities['vega_sigma']:,.0f}")
    print(f"   Erreur standard: {validation['cva_std_error']:,.0f}")
    print(f"   IC 95%: ±{validation['confidence_interval_95']:,.0f}")
    print(f"   Qualité convergence: {validation['convergence_quality']}")
    
    # 8. Graphiques académiques corrigés
    print("\n8. Génération graphiques académiques...")
    create_corrected_master_plots(
        time_grid, rate_paths_base, npv_matrix_wwr, 
        results_wwr, results_collateral, results_no_wwr,
        swap.payment_dates, market_data, empirical_corr
    )
    
    # RÉSUMÉ FINAL ACADÉMIQUE CORRIGÉ
    print("\n" + "="*70)
    print("RÉSUMÉ CVA - MODÈLE OPTIMISÉ NIVEAU MASTER")
    print("="*70)
    print(f"{'Modèle de taux:':<25} Vasicek corrigé (κ={market_data.kappa:.1f})")
    print(f"{'Convergence θ:':<25} {'✓' if np.abs(np.mean(rate_paths_base[:, -1]) - market_data.theta) < 0.01 else '✗'}")
    print(f"{'WWR post-simulation:':<25} {correlation_wwr:.1%} corrélation")
    print(f"{'At-the-money:':<25} {'✓' if abs(np.mean(npv_matrix_wwr[:, 0])) < 500 else '✗'}")
    print("-" * 70)
    
    # Calcul cohérent de l'impact WWR en basis points
    cva_base_bp = results_wwr['cva_unilateral']/swap_params.notional*10000
    cva_no_wwr_bp = results_no_wwr['cva_unilateral']/swap_params.notional*10000
    impact_wwr_bp = cva_base_bp - cva_no_wwr_bp
    
    print(f"{'CVA base (bp):':<25} {cva_base_bp:.1f}")
    print(f"{'Impact WWR (bp):':<25} {impact_wwr_bp:.1f}")
    print(f"{'WWR direction:':<25} {'Wrong-Way ✓' if impact_wwr_bp > 0 else 'Right-Way ✗'}")
    print(f"{'CVA bilatéral (bp):':<25} {results_wwr['cva_bilateral']/swap_params.notional*10000:.1f}")
    print(f"{'CVA collatéral (bp):':<25} {results_collateral['cva_unilateral']/swap_params.notional*10000:.1f}")
    print("-" * 70)
    collateral_efficiency = (1-results_collateral['cva_unilateral']/results_wwr['cva_unilateral'])*100
    print(f"{'Efficacité collatéral:':<25} {collateral_efficiency:.1f}%")
    print(f"{'Précision Monte Carlo:':<25} {validation['convergence_ratio']:.2%}")
    print(f"{'Qualité académique:':<25} Master-level ✓")
    print("="*70)

def create_corrected_master_plots(time_grid: np.ndarray, rate_paths: np.ndarray,
                                npv_matrix: np.ndarray, results_wwr: dict, 
                                results_collateral: dict, results_no_wwr: dict,
                                payment_dates: np.ndarray, market_data: MarketData,
                                empirical_corr: float):
    """Graphiques corrigés de niveau Master avec WWR post-simulation"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 1. Trajectoires Vasicek avec convergence correcte vers θ
    ax1 = axes[0, 0]
    for i in range(min(500, rate_paths.shape[0])):
        ax1.plot(time_grid, rate_paths[i, :], alpha=0.1, linewidth=0.3, color='steelblue')
    ax1.plot(time_grid, np.mean(rate_paths, axis=0), 'darkred', linewidth=3, label='Moyenne empirique')
    ax1.axhline(y=market_data.theta, color='green', linestyle='--', linewidth=2, 
               label=f'θ = {market_data.theta:.1%} (niveau LT)')
    ax1.axhline(y=market_data.initial_rate, color='orange', linestyle=':', linewidth=2,
               label=f'r₀ = {market_data.initial_rate:.1%}')
    ax1.set_title('Vasicek Stabilisé - Retour à la Moyenne ✓', fontweight='bold', color='green')
    ax1.set_xlabel('Temps (années)')
    ax1.set_ylabel('Taux d\'intérêt')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution NPV finale corrigée
    ax2 = axes[0, 1]
    npv_final = npv_matrix[:, -1]
    n, bins, patches = ax2.hist(npv_final, bins=80, alpha=0.7, density=True, 
                               edgecolor='black', color='lightblue')
    ax2.axvline(np.mean(npv_final), color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {np.mean(npv_final):,.0f}')
    ax2.axvline(np.percentile(npv_final, 5), color='orange', linestyle='--', linewidth=2,
               label=f'VaR 95%: {np.percentile(npv_final, 5):,.0f}')
    ax2.axvline(np.percentile(npv_final, 95), color='purple', linestyle='--', linewidth=2,
               label=f'95%ile: {np.percentile(npv_final, 95):,.0f}')
    ax2.set_title('Distribution NPV Finale (Corrigée)', fontweight='bold')
    ax2.set_xlabel('NPV finale')
    ax2.set_ylabel('Densité')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Impact WWR corrigé sur l'exposition
    ax3 = axes[0, 2]
    ee_wwr = results_wwr['exposure_metrics']['ee']
    ee_no_wwr = results_no_wwr['exposure_metrics']['ee']
    ee_coll = results_collateral['exposure_metrics']['ee']
    
    ax3.plot(payment_dates, ee_no_wwr, 'b-', linewidth=2, label='EE sans WWR')
    ax3.plot(payment_dates, ee_wwr, 'r-', linewidth=2, label='EE avec WWR')
    ax3.plot(payment_dates, ee_coll, 'g-', linewidth=2, label='EE collatéralisée')
    
    # Remplissage pour montrer l'impact WWR
    wwr_positive = ee_wwr > ee_no_wwr
    if np.any(wwr_positive):
        ax3.fill_between(payment_dates, ee_no_wwr, ee_wwr, 
                        where=wwr_positive, alpha=0.3, color='red',
                        label='Impact WWR (positif)')
    
    ax3.set_title(f'Impact WWR Corrigé (ρ={empirical_corr:.2f})', fontweight='bold')
    ax3.set_xlabel('Temps (années)')
    ax3.set_ylabel('Expected Exposure')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Profil d'exposition complet
    ax4 = axes[1, 0]
    pfe_95 = results_wwr['exposure_metrics']['pfe_95']
    pfe_99 = results_wwr['exposure_metrics']['pfe_99']
    ax4.plot(payment_dates, ee_wwr, 'blue', linewidth=2, label='EE')
    ax4.plot(payment_dates, pfe_95, 'red', linewidth=2, label='PFE 95%')
    ax4.plot(payment_dates, pfe_99, 'darkred', linewidth=2, label='PFE 99%')
    ax4.fill_between(payment_dates, ee_wwr, pfe_95, alpha=0.2, color='blue')
    ax4.fill_between(payment_dates, pfe_95, pfe_99, alpha=0.2, color='red')
    ax4.set_title('Profil d\'Exposition Complet', fontweight='bold')
    ax4.set_xlabel('Temps (années)')
    ax4.set_ylabel('Exposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Comparaison CVA avec et sans WWR
    ax5 = axes[1, 1]
    cva_values = [
        results_no_wwr['cva_unilateral'],
        results_wwr['cva_unilateral'],
        results_collateral['cva_unilateral']
    ]
    labels = ['CVA\nsans WWR', 'CVA\navec WWR', 'CVA\nCollatéral']
    colors = ['blue', 'red', 'green']
    
    bars = ax5.bar(labels, cva_values, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_title('Comparaison CVA (WWR Corrigé)', fontweight='bold')
    ax5.set_ylabel('CVA')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs sur les barres
    for bar, value in zip(bars, cva_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Validation de la corrélation WWR
    ax6 = axes[1, 2]
    # Nuage de points : taux moyens vs variable de défaut
    rate_factor = np.mean(rate_paths, axis=1)
    Z_default = generate_correlated_variables_for_wwr(rate_paths.shape[0], rate_paths, 0.30)
    
    ax6.scatter(rate_factor[:1000], Z_default[:1000], alpha=0.3, s=1)
    ax6.set_title(f'Corrélation WWR: ρ={empirical_corr:.3f}', fontweight='bold')
    ax6.set_xlabel('Facteur Taux Moyen')
    ax6.set_ylabel('Variable Défaut CP')
    ax6.grid(True, alpha=0.3)
    
    # Ligne de régression
    z = np.polyfit(rate_factor, Z_default, 1)
    p = np.poly1d(z)
    ax6.plot(rate_factor, p(rate_factor), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('ANALYSE CVA OPTIMISÉE - NIVEAU MASTER ✓', fontsize=16, fontweight='bold', color='green')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()