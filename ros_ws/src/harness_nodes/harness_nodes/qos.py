"""Named QoS profiles. Keep these contracts centralized and reviewable."""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

TEXT_EVENTS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

MEDIA_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

OUTPUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
