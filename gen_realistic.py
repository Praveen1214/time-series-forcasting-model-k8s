"""
Realistic Kubernetes Autoscaling Dataset Generator
Based on actual production patterns and proper metric correlations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class RealisticK8sDatasetGenerator:
    def __init__(self, start_date='2025-11-01', duration_days=30, interval_minutes=1):
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.interval_minutes = interval_minutes
        self.n_samples = int((duration_days * 24 * 60) / interval_minutes)
        self.timestamps = pd.date_range(
            start=self.start_date,
            periods=self.n_samples,
            freq=f'{interval_minutes}min'
        )
        
        # Realistic service configuration (busy e-commerce order service)
        self.config = {
            'service_capacity_rps_per_pod': 120, # Max RPS per pod before degradation
            'target_cpu_percent': 55,            # HPA target CPU (aggressive for better scaling)
            'min_pods': 2,
            'max_pods': 25,
            'scale_up_stabilization_sec': 0,     # K8s default
            'scale_down_stabilization_sec': 300, # 5 min cooldown
            'cpu_per_request': 0.65,             # Higher CPU per request = more scaling
            'base_latency_ms': 35,               # Base service latency
            'memory_base_mb': 450,               # Base memory per pod
            'memory_per_rps': 1.2,               # Memory growth per RPS
            'base_traffic': 400,                 # Higher base traffic
        }

    def generate_realistic_traffic(self):
        """Generate traffic that looks like real user behavior"""
        n = self.n_samples
        hours = self.timestamps.hour + self.timestamps.minute / 60
        day_of_week = self.timestamps.dayofweek
        day_of_month = self.timestamps.day
        
        # Base traffic using real e-commerce patterns
        base = self.config['base_traffic'] * np.ones(n)
        
        # Hourly pattern (smoother, more realistic)
        hourly = np.zeros(n)
        for i, h in enumerate(hours):
            if 0 <= h < 6:      # Night: 15-25% of peak
                hourly[i] = 0.2 + 0.05 * np.sin(h * np.pi / 6)
            elif 6 <= h < 9:    # Morning ramp: 25-70%
                hourly[i] = 0.25 + 0.45 * ((h - 6) / 3) ** 1.5
            elif 9 <= h < 12:   # Late morning: 70-90%
                hourly[i] = 0.7 + 0.2 * np.sin((h - 9) * np.pi / 6)
            elif 12 <= h < 14:  # Lunch dip: 75-85%
                hourly[i] = 0.8 - 0.05 * np.sin((h - 12) * np.pi / 2)
            elif 14 <= h < 20:  # Afternoon/evening peak: 85-100%
                hourly[i] = 0.85 + 0.15 * np.sin((h - 14) * np.pi / 12)
            elif 20 <= h < 23:  # Evening decline: 60-85%
                hourly[i] = 0.85 - 0.25 * ((h - 20) / 3)
            else:               # Late night: 30-40%
                hourly[i] = 0.35 - 0.05 * (h - 23)
        
        # Weekly pattern
        weekly = np.ones(n)
        for i, dow in enumerate(day_of_week):
            if dow == 0:    weekly[i] = 1.05   # Monday
            elif dow == 1:  weekly[i] = 1.12   # Tuesday
            elif dow == 2:  weekly[i] = 1.15   # Wednesday (peak)
            elif dow == 3:  weekly[i] = 1.10   # Thursday
            elif dow == 4:  weekly[i] = 0.95   # Friday
            elif dow == 5:  weekly[i] = 0.70   # Saturday
            else:           weekly[i] = 0.55   # Sunday
        
        # Monthly pattern (payday effects)
        monthly = np.ones(n)
        for i, dom in enumerate(day_of_month):
            if dom in [1, 2]:       monthly[i] = 1.20  # Month start
            elif dom in [14, 15, 16]: monthly[i] = 1.15  # Mid-month payday
            elif dom in [29, 30, 31]: monthly[i] = 1.25  # Month end
            elif 5 <= dom <= 10:    monthly[i] = 0.90  # Mid-month dip
        
        # Combine patterns
        traffic = base * hourly * weekly * monthly
        
        # Add realistic noise (not Gaussian - use mixture)
        # Small continuous noise + occasional larger variations
        continuous_noise = np.random.normal(0, 0.05, n)
        burst_noise = np.random.exponential(0.02, n) * np.sign(np.random.randn(n))
        traffic = traffic * (1 + continuous_noise + burst_noise)
        
        # Add organic traffic spikes (viral moments, marketing)
        traffic = self._add_organic_spikes(traffic)
        
        # Smooth to remove unrealistic jumps
        traffic = gaussian_filter1d(traffic, sigma=2)
        
        return np.maximum(10, traffic)

    def _add_organic_spikes(self, traffic):
        """Add realistic traffic spikes that build up and decay naturally"""
        n = len(traffic)
        
        # Marketing campaigns (1-2 per week, gradual build)
        n_campaigns = self.duration_days // 5
        for _ in range(n_campaigns):
            peak_idx = np.random.randint(n // 10, n - n // 10)
            # Only during business hours
            if 9 <= self.timestamps[peak_idx].hour <= 20:
                # Campaign builds over 2-4 hours, sustains, then decays
                build_time = np.random.randint(60, 180)
                sustain_time = np.random.randint(120, 360)
                decay_time = np.random.randint(180, 360)
                magnitude = np.random.uniform(1.5, 3.0)
                
                # Build phase
                for i in range(build_time):
                    idx = peak_idx - build_time + i
                    if 0 <= idx < n:
                        progress = i / build_time
                        traffic[idx] *= 1 + (magnitude - 1) * (progress ** 2)
                
                # Sustain phase
                for i in range(sustain_time):
                    idx = peak_idx + i
                    if 0 <= idx < n:
                        traffic[idx] *= magnitude * np.random.uniform(0.9, 1.1)
                
                # Decay phase
                for i in range(decay_time):
                    idx = peak_idx + sustain_time + i
                    if 0 <= idx < n:
                        progress = i / decay_time
                        traffic[idx] *= 1 + (magnitude - 1) * ((1 - progress) ** 1.5)
        
        # Flash sales (quick spike and decay)
        n_flash = self.duration_days // 3
        for _ in range(n_flash):
            peak_idx = np.random.randint(0, n)
            if 10 <= self.timestamps[peak_idx].hour <= 22:
                duration = np.random.randint(15, 45)
                magnitude = np.random.uniform(2.0, 4.0)
                
                for i in range(-duration, duration):
                    idx = peak_idx + i
                    if 0 <= idx < n:
                        dist = abs(i) / duration
                        traffic[idx] *= 1 + (magnitude - 1) * np.exp(-3 * dist ** 2)
        
        return traffic

    def simulate_hpa(self, traffic):
        """Simulate realistic HPA behavior with proper dynamics"""
        n = len(traffic)
        pod_count = np.zeros(n, dtype=int)
        cpu_usage = np.zeros(n)
        
        current_pods = self.config['min_pods']
        last_scale_up = -1000
        last_scale_down = -1000
        
        # HPA uses rolling average for stability
        cpu_history = []
        
        for i in range(n):
            # Calculate load per pod
            load_per_pod = traffic[i] / max(current_pods, 1)
            
            # CPU is proportional to load (with some base overhead)
            base_cpu = 12 + np.random.uniform(-2, 2)  # Idle CPU
            load_cpu = load_per_pod * self.config['cpu_per_request']
            
            # Add realistic CPU noise (GC, background tasks)
            cpu_noise = np.random.normal(0, 2.5)
            gc_spike = np.random.exponential(4) if np.random.random() < 0.02 else 0
            
            current_cpu = base_cpu + load_cpu + cpu_noise + gc_spike
            current_cpu = np.clip(current_cpu, 8, 95)
            cpu_usage[i] = current_cpu
            
            # HPA decision (uses average of recent CPU readings)
            cpu_history.append(current_cpu)
            if len(cpu_history) > 3:  # Faster reaction
                cpu_history.pop(0)
            avg_cpu = np.mean(cpu_history)
            
            # Scale up decision (responsive)
            if avg_cpu > self.config['target_cpu_percent']:
                if i - last_scale_up >= 1:  # Can scale up quickly
                    desired = int(np.ceil(current_pods * avg_cpu / self.config['target_cpu_percent']))
                    new_pods = min(desired, current_pods + 4, self.config['max_pods'])
                    if new_pods > current_pods:
                        current_pods = new_pods
                        last_scale_up = i
            
            # Scale down decision (conservative)
            elif avg_cpu < self.config['target_cpu_percent'] * 0.45:
                if i - last_scale_down >= 5:  # 5-minute stabilization
                    desired = max(int(np.ceil(current_pods * avg_cpu / self.config['target_cpu_percent'])),
                                 self.config['min_pods'])
                    if desired < current_pods:
                        current_pods = max(current_pods - 1, self.config['min_pods'])
                        last_scale_down = i
            
            pod_count[i] = current_pods
        
        return pod_count, cpu_usage

    def calculate_latency(self, traffic, pod_count, cpu_usage):
        """Calculate latency using queueing theory (M/M/c model approximation)"""
        n = len(traffic)
        latency = np.zeros(n)
        
        for i in range(n):
            pods = max(pod_count[i], 1)
            rps = traffic[i]
            cpu = cpu_usage[i]
            
            # Service rate per pod
            service_rate = self.config['service_capacity_rps_per_pod']
            
            # Utilization
            utilization = rps / (pods * service_rate)
            utilization = np.clip(utilization, 0.01, 0.95)
            
            # Base processing time
            base = self.config['base_latency_ms']
            
            # Queueing delay (increases non-linearly with utilization)
            queue_delay = base * (utilization ** 2) / (1 - utilization) * 0.5
            
            # CPU contention adds latency
            cpu_delay = max(0, (cpu - 70) * 0.8)
            
            # Database/downstream variance
            db_delay = np.random.gamma(2, 5)
            
            # Network jitter
            network = np.random.exponential(3)
            
            # Occasional slow requests (tail latency)
            tail = np.random.exponential(20) if np.random.random() < 0.03 else 0
            
            latency[i] = base + queue_delay + cpu_delay + db_delay + network + tail
        
        # Smooth slightly (real metrics are often aggregated)
        latency = gaussian_filter1d(latency, sigma=1)
        return np.clip(latency, 30, 500)

    def calculate_memory(self, traffic, pod_count):
        """Calculate memory usage with realistic patterns"""
        n = len(traffic)
        memory = np.zeros(n)
        
        # Memory has inertia - doesn't change instantly
        current_memory = self.config['memory_base_mb']
        
        for i in range(n):
            pods = max(pod_count[i], 1)
            rps = traffic[i]
            
            # Target memory based on load
            target = self.config['memory_base_mb'] + (rps / pods) * self.config['memory_per_rps']
            
            # Memory changes slowly (connection pools, caches)
            current_memory = 0.95 * current_memory + 0.05 * target
            
            # Add noise and GC effects
            noise = np.random.normal(0, 20)
            gc_drop = -np.random.exponential(50) if np.random.random() < 0.01 else 0
            
            memory[i] = current_memory + noise + gc_drop
        
        return np.clip(memory, 400, 1500)

    def calculate_errors(self, traffic, pod_count, cpu_usage, latency):
        """Calculate error rate based on system stress"""
        n = len(traffic)
        errors = np.zeros(n)
        
        for i in range(n):
            pods = max(pod_count[i], 1)
            load_per_pod = traffic[i] / pods
            
            # Base error rate
            base = 0.02  # 0.02% baseline
            
            # Overload errors (non-linear) - happens when load is high
            capacity = self.config['service_capacity_rps_per_pod']
            utilization = load_per_pod / capacity
            overload = max(0, (utilization - 0.7)) ** 2 * 3
            
            # CPU throttling errors
            cpu_errors = max(0, (cpu_usage[i] - 70) / 25) ** 1.5 * 0.3
            
            # High traffic naturally has more variance
            traffic_factor = (traffic[i] / self.config['base_traffic']) * 0.01
            
            # Timeout errors from high latency
            timeout_errors = max(0, (latency[i] - 100) / 80) ** 1.5 * 0.2
            
            # Random infrastructure errors
            infra = np.random.exponential(0.02) if np.random.random() < 0.01 else 0
            
            errors[i] = base + overload + cpu_errors + traffic_factor + timeout_errors + infra
        
        # Errors often come in bursts
        errors = gaussian_filter1d(errors, sigma=1.5)
        return np.clip(errors, 0.01, 8)

    def generate_dataset(self):
        """Generate complete realistic dataset"""
        print("Generating realistic K8s autoscaling dataset...")
        print(f"Duration: {self.duration_days} days, Interval: {self.interval_minutes} min")
        print(f"Samples: {self.n_samples:,}")
        print("-" * 50)
        
        # Generate traffic (the driver of everything else)
        print("1/6 Generating traffic patterns...")
        traffic = self.generate_realistic_traffic()
        
        # Simulate HPA (determines pod count and CPU)
        print("2/6 Simulating HPA behavior...")
        pod_count, cpu_avg = self.simulate_hpa(traffic)
        
        # Calculate dependent metrics
        print("3/6 Calculating latency...")
        latency_p95 = self.calculate_latency(traffic, pod_count, cpu_avg)
        
        print("4/6 Calculating memory...")
        memory_avg = self.calculate_memory(traffic, pod_count)
        
        print("5/6 Calculating error rates...")
        error_rate = self.calculate_errors(traffic, pod_count, cpu_avg, latency_p95)
        
        print("6/6 Building dataset...")
        
        # Build DataFrame with proper correlations
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'service_id': 'Order',
            'current_pod_count': pod_count,
            'request_rate_rps': np.round(traffic, 2),
            'latency_p95_ms': np.round(latency_p95, 2),
            'latency_p99_ms': np.round(latency_p95 * np.random.uniform(1.2, 1.4, self.n_samples), 2),
            'error_rate_percent': np.round(error_rate, 4),
            'queue_length': np.round(np.maximum(0, (traffic - pod_count * 70) / 30 + np.random.exponential(0.3, self.n_samples)), 2),
            'pod_cpu_usage_percent_avg': np.round(cpu_avg, 2),
            'pod_cpu_usage_percent_p95': np.round(cpu_avg * np.random.uniform(1.1, 1.25, self.n_samples), 2),
            'pod_memory_usage_mb_avg': np.round(memory_avg, 1),
            'pod_memory_usage_mb_p95': np.round(memory_avg * np.random.uniform(1.05, 1.12, self.n_samples), 1),
        })
        
        # Temporal features
        df['hour_sin'] = np.round(np.sin(2 * np.pi * df['timestamp'].dt.hour / 24), 4)
        df['hour_cos'] = np.round(np.cos(2 * np.pi * df['timestamp'].dt.hour / 24), 4)
        df['day_sin'] = np.round(np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7), 4)
        df['day_cos'] = np.round(np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7), 4)
        
        # Service mesh metrics (correlated with actual metrics)
        df['mesh_inbound_rps'] = np.round(df['request_rate_rps'] * np.random.uniform(0.98, 1.02, self.n_samples), 2)
        df['mesh_inbound_latency_p95'] = np.round(df['latency_p95_ms'] * np.random.uniform(0.92, 0.98, self.n_samples), 2)
        df['mesh_inbound_error_rate'] = np.round(df['error_rate_percent'] * np.random.uniform(0.9, 1.1, self.n_samples), 4)
        
        # Network centrality (relatively stable for a service)
        base_degree = 0.82 + 0.03 * np.sin(2 * np.pi * np.arange(self.n_samples) / (24 * 60))
        df['degree_centrality'] = np.round(base_degree + np.random.normal(0, 0.02, self.n_samples), 4)
        df['eigenvector_centrality'] = np.round(0.88 + np.random.normal(0, 0.015, self.n_samples), 4)
        df['betweenness_centrality'] = np.round(0.65 + np.random.normal(0, 0.025, self.n_samples), 4)
        df['closeness_centrality'] = np.round(0.72 + np.random.normal(0, 0.02, self.n_samples), 4)
        
        # Clip centrality metrics
        for col in ['degree_centrality', 'eigenvector_centrality', 'betweenness_centrality', 'closeness_centrality']:
            df[col] = np.clip(df[col], 0.4, 0.98)
        
        print("-" * 50)
        print("Dataset Statistics:")
        print(f"  Traffic RPS: {df['request_rate_rps'].min():.1f} - {df['request_rate_rps'].max():.1f} (mean: {df['request_rate_rps'].mean():.1f})")
        print(f"  Pod Count:   {df['current_pod_count'].min()} - {df['current_pod_count'].max()} (mean: {df['current_pod_count'].mean():.1f})")
        print(f"  CPU Usage:   {df['pod_cpu_usage_percent_avg'].min():.1f}% - {df['pod_cpu_usage_percent_avg'].max():.1f}% (mean: {df['pod_cpu_usage_percent_avg'].mean():.1f}%)")
        print(f"  Latency P95: {df['latency_p95_ms'].min():.1f} - {df['latency_p95_ms'].max():.1f} ms (mean: {df['latency_p95_ms'].mean():.1f} ms)")
        print(f"  Error Rate:  {df['error_rate_percent'].min():.3f}% - {df['error_rate_percent'].max():.3f}% (mean: {df['error_rate_percent'].mean():.3f}%)")
        
        return df

    def validate_correlations(self, df):
        """Check that metric correlations are realistic"""
        print("\nMetric Correlations (should match real K8s patterns):")
        print("-" * 50)
        
        corr_pairs = [
            ('request_rate_rps', 'pod_cpu_usage_percent_avg', 'RPS vs CPU', 0.5, 0.9),
            ('request_rate_rps', 'current_pod_count', 'RPS vs Pods', 0.4, 0.85),
            ('pod_cpu_usage_percent_avg', 'latency_p95_ms', 'CPU vs Latency', 0.2, 0.7),
            ('request_rate_rps', 'error_rate_percent', 'RPS vs Errors', 0.1, 0.5),
        ]
        
        for col1, col2, name, low, high in corr_pairs:
            corr = df[col1].corr(df[col2])
            status = "OK" if low <= corr <= high else "CHECK"
            print(f"  {name}: {corr:.3f} (expected: {low:.1f}-{high:.1f}) [{status}]")


if __name__ == "__main__":
    generator = RealisticK8sDatasetGenerator(
        start_date='2025-11-01',
        duration_days=30,
        interval_minutes=1
    )
    
    df = generator.generate_dataset()
    generator.validate_correlations(df)
    
    # Save
    filename = 'data.csv'
    df.to_csv(filename, index=False)
    print(f"\nSaved to: {filename}")
    
    # Also save to notebooks folder
    notebook_path = 'notebooks/Informer-model-direct-podcount-prediction/data.csv'
    df.to_csv(notebook_path, index=False)
    print(f"Saved to: {notebook_path}")
