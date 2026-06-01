from enum import Enum, auto, IntEnum
from dataclasses import dataclass
from flight_frame import FlightFrame
import math 

class Field(Enum):
    ALTITUDE = auto()
    AOA = auto()
    GLoad = auto()
    IAS = auto()
    AoB = auto()
    VS = auto() #vertical speed

class Comparison(Enum):
    LESS_THAN = auto()
    GREATER_THAN = auto()

class Severity(IntEnum):
    NORMAL = auto()
    CAUTIOUS = auto() 
    CRITICAL = auto()
    CATASTROPHIC = auto()

class FlightStage (Enum):
    TAXI = auto()
    TAKE_OFF = auto()
    CRUISE = auto()
    CLIMB = auto()
    DESCEND = auto()
    LANDING = auto()

class TargetField (Enum):
    INCREMENT = auto()
    DECREMENT = auto()

class Urgency (IntEnum): 
    LOW = auto()
    MID = auto()
    HIGH = auto()

# Threshold value: violations
MIN_ALT = 100.0
MAX_ALT = 35000.0
MIN_AOA = -7.0
MAX_AOA = 35.0
MAX_GLOAD = 7.0
MIN_GLOAD = -2.0
MIN_IAS = 140.0
MAX_IAS = 700.0
MAX_AOB = 75.0
MAX_VS = 8.0       #CRUISE ONLY
MIN_VS = -8.0      #CRUISE ONLY

#for phase
GEAR_UP_THRES = 0.05
GEAR_DOWN_THRES = 0.95
LIFT_OFF_ALT = 3.0
MAX_CRUISE_VS = 2.0
MIN_CRUISE_VS = -2.0
MAX_TAXI_SPEED = 15.0

#for severity: Amount below are EXCEEDING part from the MAX and MIN threshold
CAUTIOUS_ALT = 30.0
CRITICAL_ALT = 50.0
CAUT_MAX_ALT = 1000.0
CRIT_MAX_ALT = 3000.0
CAUT_MAX_AOA = 5.0
CRIT_MAX_AOA = 10.0
CAUT_MIN_AOA = 2.0
CRIT_MIN_AOA = 4.0
CAUT_MAX_G = 0.5
CRIT_MAX_G = 1.0
CAUT_MIN_G = 0.5
CRIT_MIN_G = 1.0
CAUT_MIN_IAS = 10.0
CRIT_MIN_IAS = 20.0
CAUT_MAX_IAS = 100.0
CRIT_MAX_IAS = 200.0
CAUT_MAX_AOB = 15.0
CRIT_MAX_AOB = 45.0
CAUT_MAX_VS  = 2.0
CRIT_MAX_VS  = 5.0
CAUT_MIN_VS  = 2.0
CRIT_MIN_VS  = 5.0

@dataclass
class NatopsRule:
    name: str
    field: Field
    comparison: Comparison
    threshold: float 
    cautious_pct: float
    critical_pct: float
    severity: Severity
    phases: list[FlightStage]

min_alt = NatopsRule (
    name = "Minimum Altitude",
    field = Field.ALTITUDE,
    comparison = Comparison.LESS_THAN,
    threshold = MIN_ALT,
    cautious_pct = CAUTIOUS_ALT,
    critical_pct = CRITICAL_ALT,
    severity = Severity.CRITICAL,
    phases = [FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND],
)

max_alt = NatopsRule (
    name = "Maximum Altitude",
    field = Field.ALTITUDE,
    comparison = Comparison.GREATER_THAN,
    threshold = MAX_ALT, 
    cautious_pct = CAUT_MAX_ALT,
    critical_pct = CRIT_MAX_ALT,
    severity = Severity.CAUTIOUS,
    phases = [FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND],
)

max_aoa = NatopsRule (
    name = "Max AoA",
    field = Field.AOA,
    comparison = Comparison.GREATER_THAN,
    threshold = MAX_AOA,
    cautious_pct = CAUT_MAX_AOA,
    critical_pct = CRIT_MAX_AOA,
    severity = Severity.CRITICAL,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

min_aoa = NatopsRule (
    name = "Min AoA",
    field = Field.AOA,
    comparison = Comparison.LESS_THAN,
    threshold = MIN_AOA,
    cautious_pct = CAUT_MIN_AOA,
    critical_pct = CRIT_MIN_AOA,
    severity = Severity.CRITICAL,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

max_pos_g = NatopsRule (
    name = "Max Positive G",
    field = Field.GLoad,
    comparison = Comparison.GREATER_THAN,
    threshold = MAX_GLOAD,
    cautious_pct = CAUT_MAX_G,
    critical_pct = CRIT_MAX_G,
    severity = Severity.CRITICAL,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

max_neg_g = NatopsRule (
    name = "Max Negative G",
    field = Field.GLoad,
    comparison = Comparison.LESS_THAN,
    threshold = MIN_GLOAD,
    cautious_pct = CAUT_MIN_G,
    critical_pct = CRIT_MIN_G,
    severity = Severity.CRITICAL,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

max_speed = NatopsRule (
    name = "Max Speed",
    field = Field.IAS,
    comparison = Comparison.GREATER_THAN,
    threshold = MAX_IAS,
    cautious_pct = CAUT_MAX_IAS,
    critical_pct = CRIT_MAX_IAS,
    severity = Severity.CRITICAL,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

stall_speed = NatopsRule (
    name = "Stall Speed",
    field = Field.IAS,
    comparison = Comparison.LESS_THAN,
    threshold = MIN_IAS,
    cautious_pct = CAUT_MIN_IAS,
    critical_pct = CRIT_MIN_IAS,
    severity = Severity.CRITICAL,
    phases = [FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND]
)

max_aob = NatopsRule (
    name = "Max AoB",
    field = Field.AoB,
    comparison = Comparison.GREATER_THAN,
    threshold = MAX_AOB,
    cautious_pct = CAUT_MAX_AOB,
    critical_pct = CRIT_MAX_AOB,
    severity = Severity.CAUTIOUS,
    phases = [FlightStage.TAKE_OFF, FlightStage.CLIMB, FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.LANDING]
)

max_vs = NatopsRule (   #CRUISE ONLY vertical speed rule
    name = "Max VS",
    field = Field.VS,
    comparison= Comparison.GREATER_THAN,
    threshold = MAX_VS,
    cautious_pct=CAUT_MAX_VS,
    critical_pct=CRIT_MAX_VS,
    severity=Severity.CRITICAL,
    phases = [FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.CLIMB]
)
min_vs = NatopsRule (   #CRUISE ONLY vertical speed rule
    name = "Min VS",
    field = Field.VS,
    comparison= Comparison.LESS_THAN,
    threshold = MIN_VS,
    cautious_pct=CAUT_MIN_VS,
    critical_pct=CRIT_MIN_VS,
    severity=Severity.CRITICAL,
    phases = [FlightStage.CRUISE, FlightStage.DESCEND, FlightStage.CLIMB]
)

f18_rules: list[NatopsRule] = [
    min_alt, max_alt, max_aoa, min_aoa, max_pos_g, max_neg_g, max_speed, stall_speed, max_aob, max_vs, min_vs
]
#return converted value for precise 
def get_value(frame: FlightFrame, field: Field) -> float:
    if field == Field.ALTITUDE:
        return frame.alt_msl_m * 3.28084
    elif field == Field.AOA:
        return frame.aoa_rad
    elif field == Field.GLoad:
        return frame.g_load
    elif field == Field.IAS:
        return frame.ias_ms * 1.94384
    elif field ==Field.VS:
        return frame.vertical_speed_ms
    else: # field == Field.AOB
        return frame.bank_rad * 57.29578

#how serious are each violation episode
def compute_severity(rule: NatopsRule, off_value: float) -> Severity:
    delta = abs(off_value - rule.threshold)
    if delta <= rule.cautious_pct:
        return Severity.CAUTIOUS
    elif delta > rule.cautious_pct and delta <= rule.critical_pct:
        return Severity.CRITICAL
    else: # delta > rule.critical_pct
        return Severity.CATASTROPHIC
    
def is_violated(rule: NatopsRule, frame: FlightFrame, phase: FlightStage) -> bool: 
    if phase not in rule.phases:
        return False
    result = get_value(frame, rule.field)

    if rule.comparison == Comparison.LESS_THAN:
        return result < rule.threshold
    elif rule.comparison == Comparison.GREATER_THAN:
        return result > rule.threshold
    else:
        return False

def detect_phase(frame: FlightFrame) -> FlightStage:
    #Gear position check
    ground_speed = math.sqrt(frame.vx_ms * frame.vx_ms + frame.vz_ms * frame.vz_ms)
    gear_down: bool = frame.gear_pos > GEAR_DOWN_THRES        #gear down if gear_pos > 0.95
    gear_up: bool = frame.gear_pos < GEAR_UP_THRES
    on_ground: bool = frame.alt_agl_m < LIFT_OFF_ALT

    #Gear Down Circumstances
    is_taxi: bool = on_ground and ground_speed < MAX_TAXI_SPEED;
    is_ground_roll: bool = on_ground and ground_speed >= MAX_TAXI_SPEED;   #first kind of take off
    is_goAround: bool = not on_ground and frame.vertical_speed_ms > 0;    #second kind of take off
    is_landing: bool = not on_ground and frame.vertical_speed_ms <= MIN_CRUISE_VS; 

    #Gear Up Circumstances
    is_cruising: bool = frame.vertical_speed_ms >= MIN_CRUISE_VS and frame.vertical_speed_ms <= MAX_CRUISE_VS;
    is_climbing: bool = frame.vertical_speed_ms > MAX_CRUISE_VS;    # > 2.0
    is_descending: bool = frame.vertical_speed_ms < MIN_CRUISE_VS;  # < -2.0

    if gear_down:   #Taxi, Landing, TakeOff
        if is_taxi:
            return FlightStage.TAXI
        elif is_ground_roll: 
            return FlightStage.TAKE_OFF
        elif is_goAround: 
            return FlightStage.TAKE_OFF
        elif is_landing: 
            return FlightStage.LANDING
    elif gear_up:
        if is_cruising:
            return FlightStage.CRUISE
        elif is_climbing:
            return FlightStage.CLIMB
        elif is_descending:
            return FlightStage.DESCEND
    else: #gear in transition
        if frame.vertical_speed_ms > 0:
            return FlightStage.TAKE_OFF
        else:
            return FlightStage.LANDING
        
    return FlightStage.CRUISE





'''
    GEAR_UP_THRES = 0.05
    GEAR_DOWN_THRES = 0.95
    LIFT_OFF_ALT = 3.0
    MAX_CRUISE_VS = 2.0
    MIN_CRUISE_VS = -2.0
    MAX_TAXI_SPEED = 15.0
    phase = FlightStage.CRUISE
    print(phase.name) will just print "CRUISE"
'''