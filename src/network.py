from models import Tier

class NetworkChannel:
    """Models network links between Mist, Fog, and Cloud tiers."""
    
    # Tier-to-Tier Link Specifications: (Bandwidth in Mbps, Propagation Latency in Seconds)
    LINKS = {
        (Tier.MIST, Tier.FOG): (30.0, 0.015),
        (Tier.FOG, Tier.MIST): (30.0, 0.015),
        (Tier.FOG, Tier.CLOUD): (250.0, 0.080),
        (Tier.CLOUD, Tier.FOG): (250.0, 0.080),
        (Tier.MIST, Tier.CLOUD): (25.0, 0.100),
        (Tier.CLOUD, Tier.MIST): (25.0, 0.100),
    }

    # Transmission power in Watts
    TX_POWER = {
        Tier.MIST: 0.6,
        Tier.FOG: 2.5,
        Tier.CLOUD: 12.0
    }

    @classmethod
    def get_transfer_time(cls, src: Tier, dst: Tier, data_size_mb: float) -> float:
        """Returns the total delay (seconds) to transfer data between tiers."""
        if src == dst:
            return 0.0
        
        bw_mbps, prop_delay = cls.LINKS.get((src, dst), (20.0, 0.050))
        # data_size in megabytes -> megabits = data_size * 8
        transmission_delay = (data_size_mb * 8.0) / bw_mbps
        return transmission_delay + prop_delay

    @classmethod
    def get_transfer_energy(cls, src: Tier, dst: Tier, data_size_mb: float) -> float:
        """Calculates energy in Joules consumed during data transmission."""
        tx_time = cls.get_transfer_time(src, dst, data_size_mb)
        power_w = cls.TX_POWER.get(src, 1.0)
        return tx_time * power_w
