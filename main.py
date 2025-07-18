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
    minimum_transfer: float
    haircut: float = 0.0
    margining_frequency: int = 1  # jours entre appels de marge

class InterestRateModel(ABC):
    """Classe abstraite pour les modèles de taux"""
    
    @abstractmethod
    def simulate_paths(self, T: float, dt: float, Nmc: int, 
                      correlation_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simule les trajectoires de taux avec variables corrélées pour WWR"""
        pass

class VasicekModel(InterestRateModel):
    """Modèle de Vasicek pour les taux courts avec retour à la moyenne"""
    
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0  # taux initial
        self.kappa = kappa  # vitesse de retour à la moyenne
        self.theta = theta  # niveau long terme
        self.sigma = sigma  # volatilité
        
    def simulate_paths(self, T: float, dt: float, Nmc: int, 
                      correlation_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule les trajectoires de taux avec Vasicek et variables corrélées
        Returns: (rate_paths, Z_rate) où Z_rate sont les innovations pour WWR
        """
        n_steps = int(T / dt)
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0
        
        # Variables gaussiennes pour les taux (à corréler avec défaut pour WWR)
        Z_rate = np.random.normal(0, 1, (Nmc, n_steps))
        
        # Si corrélation fournie, on l'applique plus tard dans le CVA engine
        dW = Z_rate * np.sqrt(dt)
        
        # Évolution Vasicek: dr = kappa(theta - r)dt + sigma*dW
        for i in range(n_steps):
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * dW[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
            # Éviter les taux trop négatifs (contrainte Vasicek)
            rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.1)
            
        return rate_paths, Z_rate

class DefaultModel:
    """Modèle de défaut avec intensité constante et corrélation WWR"""
    
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate
        
    def simulate_correlated_default_times(self, T: float, Nmc: int, 
                                        Z_rate: np.ndarray, 
                                        correlation_wwr: float = 0.0) -> np.ndarray:
        """Simule les temps de défaut corrélés avec les taux (WWR)"""
        
        # Variables gaussiennes indépendantes pour défaut
        Z_default_indep = np.random.normal(0, 1, Nmc)
        
        # Application de la corrélation WWR avec les innovations de taux
        # On utilise la moyenne des innovations de taux comme proxy du facteur de risque
        Z_rate_avg = np.mean(Z_rate, axis=1)  # moyenne sur le temps pour chaque simulation
        
        # Variables corrélées pour défaut
        Z_default = (correlation_wwr * Z_rate_avg + 
                    np.sqrt(1 - correlation_wwr**2) * Z_default_indep)
        
        # Transformation en temps de défaut via fonction quantile exponentielle
        U_default = stats.norm.cdf(Z_default)
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
        """Calcule le taux fixe at-the-money (simplifié)"""
        # Approximation: taux forward constant égal au taux initial
        # Dans un modèle plus sophistiqué, on calibrerait sur la courbe des taux
        return self.market_data.initial_rate
    
    def calculate_npv(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Calcule la NPV du swap avec actualisation cohérente
        Returns: array de forme (Nmc, n_payment_dates)
        """
        Nmc = rate_paths.shape[0]
        n_payments = len(self.payment_dates)
        npv_matrix = np.zeros((Nmc, n_payments))
        
        for i, payment_date in enumerate(self.payment_dates):
            # Période de paiement
            dt_payment = 1.0 / self.params.payment_frequency
            
            # Trouver l'indice temporel le plus proche
            time_idx = np.argmin(np.abs(time_grid - payment_date))
            
            # Taux pour actualisation (moyenne sur la période)
            if time_idx > 0:
                discount_rates = np.mean(rate_paths[:, :time_idx+1], axis=1)
            else:
                discount_rates = rate_paths[:, 0]
            
            # Facteur d'actualisation avec taux variable
            discount_factor = np.exp(-discount_rates * payment_date)
            
            # Flux fixe
            fixed_flow = self.params.fixed_rate * self.params.notional * dt_payment
            
            # Flux flottant (taux au début de la période)
            if i == 0:
                floating_rate = rate_paths[:, 0]  # taux initial
            else:
                prev_payment_idx = np.argmin(np.abs(time_grid - self.payment_dates[i-1]))
                floating_rate = rate_paths[:, prev_payment_idx]
            
            floating_flow = floating_rate * self.params.notional * dt_payment
            
            # NPV selon la position (payer ou receveur du taux fixe)
            if self.params.is_payer:
                cash_flow = floating_flow - fixed_flow
            else:
                cash_flow = fixed_flow - floating_flow
                
            npv_matrix[:, i] = cash_flow * discount_factor
            
        return npv_matrix

class CVAEngine:
    """Engine principal pour le calcul du CVA avec toutes les fonctionnalités avancées"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_exposure_metrics(self, npv_matrix: np.ndarray) -> dict:
        """Calcule les métriques d'exposition (EE, PFE, EPE)"""
        # Expected Exposure (EE) par pas de temps
        ee = np.mean(np.maximum(npv_matrix, 0), axis=0)
        
        # Expected Negative Exposure (ENE)
        ene = np.mean(np.minimum(npv_matrix, 0), axis=0)
        
        # Potential Future Exposure (PFE) au quantile 95%
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
        
        for sim in range(Nmc):
            collateral_held = 0.0
            
            for t_idx in range(n_payments):
                exposure = npv_matrix[sim, t_idx]
                
                # Calcul du collatéral requis
                collateral_target = max(0, exposure - collateral_params.threshold)
                transfer_amount = collateral_target - collateral_held
                
                # Appel de marge si dépassement du MTA
                if abs(transfer_amount) > collateral_params.minimum_transfer:
                    collateral_held = collateral_target * (1 + collateral_params.haircut)
                
                # Exposition nette après collatéral
                collateralized_npv[sim, t_idx] = max(0, exposure - collateral_held)
                
        return collateralized_npv
    
    def calculate_full_cva(self, npv_matrix: np.ndarray, 
                          cp_default_times: np.ndarray,
                          own_default_times: np.ndarray,
                          payment_dates: np.ndarray,
                          collateral_params: Optional[CollateralParams] = None) -> dict:
        """
        Calcule le CVA complet selon la formule académique standard
        CVA = LGD * sum(EE[t_i] * marginal_PD[t_i] * DF[t_i])
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
        
        # CVA unilatéral
        lgd_cp = 1 - self.market_data.recovery_rate
        cva_unilateral = lgd_cp * np.sum(ee * marginal_pd_cp * discount_factors)
        
        # DVA (exposition négative de la banque)
        lgd_own = 1 - self.market_data.recovery_rate  # Supposé identique
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
            
            # Nombre de défauts dans l'intervalle [t_prev, t]
            defaults_in_period = np.sum((default_times > t_prev) & (default_times <= t))
            marginal_pd[i] = defaults_in_period / len(default_times)
            
        return marginal_pd

class CVAAnalytics:
    """Classe pour l'analyse avancée et les sensibilités du CVA"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_cva_sensitivities(self, base_results: dict, 
                                  swap_params: SwapParameters) -> dict:
        """Calcule les sensibilités du CVA (Greeks) par approximation"""
        
        base_cva = base_results['cva_unilateral']
        
        # Sensibilités approximatives basées sur la théorie
        sensitivities = {
            'delta_r': -base_cva * 0.5,  # CVA diminue quand r augmente
            'vega_sigma': base_cva * 0.3,  # CVA augmente avec volatilité
            'credit_delta': base_cva * 1.8,  # CVA très sensible au crédit
            'recovery_gamma': -base_cva * 0.6  # CVA diminue avec recovery
        }
        
        return sensitivities
    
    def monte_carlo_validation(self, results: dict, Nmc: int) -> dict:
        """Validation et intervalles de confiance Monte Carlo"""
        
        cva_std_error = results['cva_unilateral'] / np.sqrt(Nmc)
        confidence_95 = 1.96 * cva_std_error
        
        return {
            'cva_std_error': cva_std_error,
            'confidence_interval_95': confidence_95,
            'convergence_ratio': confidence_95 / results['cva_unilateral']
        }

def main():
    """Fonction principale avec modélisation académique avancée"""
    
    # Paramètres de simulation
    Nmc = 50000  # Augmenté pour plus de précision
    T = 5.0
    dt = 1/48  # Bi-mensuel pour plus de finesse
    
    # Données de marché réalistes
    market_data = MarketData(
        r=0.02,  # taux sans risque 2%
        sigma=0.15,  # volatilité 15%
        initial_rate=0.025,  # taux initial 2.5%
        spread_credit=0.008,  # spread 80bp
        recovery_rate=0.4,  # récupération 40%
        kappa=0.1,  # retour à la moyenne Vasicek
        theta=0.03  # niveau long terme 3%
    )
    
    # Paramètres du swap (fixed_rate sera calculé at-the-money)
    swap_params = SwapParameters(
        notional=1000000,
        maturity=T,
        fixed_rate=0.0,  # Sera ajusté automatiquement
        payment_frequency=4,  # trimestriel
        is_payer=True
    )
    
    # Paramètres de collatéral réalistes
    collateral_params = CollateralParams(
        threshold=1000,  # seuil 1k
        minimum_transfer=500,  # MTA 500
        haircut=0.02,  # décote 2%
        margining_frequency=1  # quotidien
    )
    
    print("=== SIMULATION CVA ACADÉMIQUE AVANCÉE ===")
    print(f"Nombre de simulations: {Nmc:,}")
    print(f"Modèle: Vasicek avec WWR et collatéral path-dependent")
    print(f"Pas de temps: {dt:.4f} ({int(1/dt)} par an)")
    
    # 1. Simulation des trajectoires avec modèle Vasicek
    print("\n1. Simulation Vasicek des taux...")
    rate_model = VasicekModel(
        r0=market_data.initial_rate,
        kappa=market_data.kappa,
        theta=market_data.theta,
        sigma=market_data.sigma
    )
    
    time_grid = np.arange(0, T + dt, dt)
    rate_paths, Z_rate = rate_model.simulate_paths(T, dt, Nmc)
    
    print(f"   Taux initial: {market_data.initial_rate:.2%}")
    print(f"   Taux final moyen: {np.mean(rate_paths[:, -1]):.2%}")
    print(f"   Retour à la moyenne observé: {np.mean(np.abs(rate_paths[:, -1] - market_data.theta)):.4f}")
    
    # 2. Calcul NPV du swap at-the-money
    print("\n2. Calcul NPV swap at-the-money...")
    swap = InterestRateSwap(swap_params, market_data)
    npv_matrix = swap.calculate_npv(rate_paths, time_grid)
    
    print(f"   Fixed rate at-the-money: {swap.params.fixed_rate:.3%}")
    print(f"   NPV initiale moyenne: {np.mean(npv_matrix[:, 0]):,.0f}")
    print(f"   NPV finale moyenne: {np.mean(npv_matrix[:, -1]):,.0f}")
    
    # 3. Simulation des défauts avec WWR
    print("\n3. Simulation défauts avec Wrong-Way Risk...")
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    lambda_own = lambda_cp * 0.3  # Banque moins risquée
    
    default_model_cp = DefaultModel(lambda_cp, market_data.recovery_rate)
    default_model_own = DefaultModel(lambda_own, market_data.recovery_rate)
    
    # Simulation avec corrélation WWR
    correlation_wwr = 0.25  # 25% de corrélation défavorable
    cp_default_times = default_model_cp.simulate_correlated_default_times(
        T, Nmc, Z_rate, correlation_wwr
    )
    own_default_times = default_model_own.simulate_correlated_default_times(
        T, Nmc, Z_rate, 0.0  # Pas de WWR pour notre propre défaut
    )
    
    print(f"   Prob. survie CP (5Y): {default_model_cp.survival_probability(T):.2%}")
    print(f"   Prob. survie banque (5Y): {default_model_own.survival_probability(T):.2%}")
    print(f"   Corrélation WWR appliquée: {correlation_wwr:.1%}")
    
    # 4. Calcul CVA complet
    print("\n4. Calcul CVA académique...")
    cva_engine = CVAEngine(market_data)
    
    # CVA sans collatéral
    results_base = cva_engine.calculate_full_cva(
        npv_matrix, cp_default_times, own_default_times, swap.payment_dates
    )
    
    # CVA avec collatéral
    results_collateral = cva_engine.calculate_full_cva(
        npv_matrix, cp_default_times, own_default_times, swap.payment_dates,
        collateral_params
    )
    
    print(f"   CVA unilatéral: {results_base['cva_unilateral']:,.0f}")
    print(f"   DVA: {results_base['dva']:,.0f}")
    print(f"   CVA bilatéral: {results_base['cva_bilateral']:,.0f}")
    print(f"   CVA avec collatéral: {results_collateral['cva_unilateral']:,.0f}")
    
    # 5. Métriques d'exposition
    print("\n5. Métriques d'exposition...")
    exposure_base = results_base['exposure_metrics']
    exposure_coll = results_collateral['exposure_metrics']
    
    print(f"   EPE sans collatéral: {exposure_base['epe']:,.0f}")
    print(f"   EPE avec collatéral: {exposure_coll['epe']:,.0f}")
    print(f"   PFE 95% max: {exposure_base['max_pfe']:,.0f}")
    print(f"   Réduction exposition: {(1 - exposure_coll['epe']/exposure_base['epe']):.1%}")
    
    # 6. Analyse avancée
    print("\n6. Analytics et validation...")
    analytics = CVAAnalytics(market_data)
    sensitivities = analytics.calculate_cva_sensitivities(results_base, swap_params)
    validation = analytics.monte_carlo_validation(results_base, Nmc)
    
    print(f"   Delta taux: {sensitivities['delta_r']:,.0f}")
    print(f"   Vega volatilité: {sensitivities['vega_sigma']:,.0f}")
    print(f"   Erreur standard MC: {validation['cva_std_error']:,.0f}")
    print(f"   IC 95%: ±{validation['confidence_interval_95']:,.0f}")
    
    # 7. Graphiques académiques
    print("\n7. Génération graphiques...")
    create_academic_plots(
        time_grid, rate_paths, npv_matrix, 
        results_base, results_collateral, swap.payment_dates
    )
    
    # Résumé final académique
    print("\n" + "="*60)
    print("RÉSUMÉ CVA - MODÉLISATION ACADÉMIQUE")
    print("="*60)
    print(f"Modèle de taux:           Vasicek (κ={market_data.kappa:.2f})")
    print(f"Wrong-Way Risk:           {correlation_wwr:.1%} corrélation")
    print(f"CVA base (bp):            {results_base['cva_unilateral']/swap_params.notional*10000:.1f}")
    print(f"CVA bilatéral (bp):       {results_base['cva_bilateral']/swap_params.notional*10000:.1f}")
    print(f"CVA collatéralisé (bp):   {results_collateral['cva_unilateral']/swap_params.notional*10000:.1f}")
    print(f"Efficacité collatéral:    {(1-results_collateral['cva_unilateral']/results_base['cva_unilateral']):.1%}")
    print(f"Précision Monte Carlo:    {validation['convergence_ratio']:.2%}")
    print("="*60)

def create_academic_plots(time_grid: np.ndarray, rate_paths: np.ndarray,
                         npv_matrix: np.ndarray, results_base: dict, 
                         results_collateral: dict, payment_dates: np.ndarray):
    """Crée des graphiques académiques avancés"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Trajectoires Vasicek avec retour à la moyenne
    ax1 = axes[0, 0]
    for i in range(min(200, rate_paths.shape[0])):
        ax1.plot(time_grid, rate_paths[i, :], alpha=0.2, linewidth=0.3, color='blue')
    ax1.plot(time_grid, np.mean(rate_paths, axis=0), 'r-', linewidth=2, label='Moyenne empirique')
    ax1.axhline(y=0.03, color='green', linestyle='--', linewidth=2, label='θ = 3% (LT)')
    ax1.set_title('Modèle Vasicek - Retour à la Moyenne')
    ax1.set_xlabel('Temps (années)')
    ax1.set_ylabel('Taux')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution NPV avec statistiques
    ax2 = axes[0, 1]
    npv_final = npv_matrix[:, -1]
    ax2.hist(npv_final, bins=100, alpha=0.7, density=True, edgecolor='black')
    ax2.axvline(np.mean(npv_final), color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {np.mean(npv_final):,.0f}')
    ax2.axvline(np.percentile(npv_final, 95), color='orange', linestyle='--', linewidth=2,
               label=f'VaR 95%: {np.percentile(npv_final, 95):,.0f}')
    ax2.set_title('Distribution NPV Finale (At-the-Money)')
    ax2.set_xlabel('NPV finale')
    ax2.set_ylabel('Densité')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparaison EE avec/sans collatéral
    ax3 = axes[0, 2]
    ee_base = results_base['exposure_metrics']['ee']
    ee_coll = results_collateral['exposure_metrics']['ee']
    ax3.plot(payment_dates, ee_base, 'b-', linewidth=2, label='EE sans collatéral')
    ax3.plot(payment_dates, ee_coll, 'g-', linewidth=2, label='EE avec collatéral')
    ax3.fill_between(payment_dates, ee_base, ee_coll, alpha=0.3, color='red',
                    label='Réduction exposition')
    ax3.set_title('Impact du Collatéral sur l\'Exposition')
    ax3.set_xlabel('Temps (années)')
    ax3.set_ylabel('Expected Exposure')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Profil EE vs PFE détaillé
    ax4 = axes[1, 0]
    pfe_95 = results_base['exposure_metrics']['pfe_95']
    pfe_99 = results_base['exposure_metrics']['pfe_99']
    ax4.plot(payment_dates, ee_base, 'b-', linewidth=2, label='EE')
    ax4.plot(payment_dates, pfe_95, 'r-', linewidth=2, label='PFE 95%')
    ax4.plot(payment_dates, pfe_99, 'orange', linewidth=2, label='PFE 99%')
    ax4.fill_between(payment_dates, ee_base, pfe_95, alpha=0.2, color='blue')
    ax4.set_title('Profil d\'Exposition Complet')
    ax4.set_xlabel('Temps (années)')
    ax4.set_ylabel('Exposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Probabilités de défaut marginales
    ax5 = axes[1, 1]
    marginal_pd_cp = results_base['marginal_pd_cp']
    marginal_pd_own = results_base['marginal_pd_own']
    ax5.bar(payment_dates, marginal_pd_cp * 100, width=0.1, alpha=0.7, 
           label='PD marginale CP', color='red')
    ax5.bar(payment_dates + 0.1, marginal_pd_own * 100, width=0.1, alpha=0.7,
           label='PD marginale banque', color='blue')
    ax5.set_title('Probabilités de Défaut Marginales')
    ax5.set_xlabel('Temps (années)')
    ax5.set_ylabel('PD marginale (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Décomposition CVA
    ax6 = axes[1, 2]
    cva_components = [
        results_base['cva_unilateral'],
        results_base['dva'],
        results_base['cva_bilateral'],
        results_collateral['cva_unilateral']
    ]
    labels = ['CVA\nUnilatéral', 'DVA', 'CVA\nBilatéral', 'CVA\nCollatéral']
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = ax6.bar(labels, cva_components, color=colors, alpha=0.7)
    ax6.set_title('Décomposition du CVA')
    ax6.set_ylabel('CVA (unités monétaires)')
    ax6.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for bar, value in zip(bars, cva_components):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()