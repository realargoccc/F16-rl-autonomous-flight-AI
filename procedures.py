from enum import Enum, auto
from natops import FlightStage, Severity
from dataclasses import dataclass, field


class Target_Field(Enum):   
    ALTITUDE = auto()
    AOA = auto() 
    GLOAD = auto()
    IAS = auto()
    AOB = auto()
    VS  = auto()

class Adjustment (Enum):    #Direction of adjustment
    INCREMENT = auto()
    DECREMENT = auto()

class Urgency (Enum):
    LOW = auto()
    MID = auto()
    HIGH = auto()

@dataclass
class Procedures:
    name: str
    field: Target_Field
    direction: Adjustment
    urgencylvl: Urgency
    target_by_phase: dict[FlightStage, float] = field(default_factory=dict)

min_alt_proc = Procedures (
    name = "Minimum Altitude",
    field = Target_Field.ALTITUDE,
    direction =  Adjustment.INCREMENT,
    urgencylvl = Urgency.HIGH,
    target_by_phase = {
        FlightStage.CRUISE: 300.0,
        FlightStage.DESCEND: 300.0,
        FlightStage.CLIMB: 300.0
    }
)

max_alt_proc = Procedures (
    name = "Maximum Altitude",
    field = Target_Field.ALTITUDE,
    direction =  Adjustment.DECREMENT,
    urgencylvl = Urgency.LOW,
    target_by_phase = {
        FlightStage.CRUISE: 30000.0,
        FlightStage.DESCEND: 30000.0,
        FlightStage.CLIMB: 30000.0
    }
)

max_aoa_proc = Procedures (
    name = "Maximum AoA",
    field = Target_Field.AOA,
    direction =  Adjustment.DECREMENT,
    urgencylvl = Urgency.HIGH,
    target_by_phase = {
        FlightStage.CRUISE: 3.0,
        FlightStage.DESCEND: 2.0,
        FlightStage.CLIMB: 5.0,
        FlightStage.TAKE_OFF: 8.0,
        FlightStage.LANDING: 8.0
    }
)

min_aoa_proc = Procedures(
    name = "Min AoA",
    field = Target_Field.AOA,
    direction = Adjustment.INCREMENT,
    urgencylvl = Urgency.MID,
    target_by_phase = {
        FlightStage.TAKE_OFF: 8.0,
        FlightStage.CLIMB: 5.0,
        FlightStage.CRUISE: 3.0,
        FlightStage.DESCEND: 2.0,
        FlightStage.LANDING: 8.0,
    },
)

max_pos_g_proc = Procedures(
    name = "Max Positive G",
    field = Target_Field.GLOAD,
    direction = Adjustment.DECREMENT,
    urgencylvl = Urgency.HIGH,
    # no target_by_phase — follow rule envelope toward ideal (1.0g)
)

max_neg_g_proc = Procedures(
    name = "Max Negative G",
    field = Target_Field.GLOAD,
    direction = Adjustment.INCREMENT,
    urgencylvl = Urgency.HIGH,
    # no target_by_phase — follow rule envelope toward ideal (1.0g)
)

max_speed_proc = Procedures(
    name = "Max Speed",
    field = Target_Field.IAS,
    direction = Adjustment.DECREMENT,
    urgencylvl = Urgency.LOW,
    target_by_phase = {
        FlightStage.CLIMB: 500.0,
        FlightStage.CRUISE: 500.0,
        FlightStage.DESCEND: 500.0,
    },
)

stall_speed_proc = Procedures(
    name = "Stall Speed",
    field = Target_Field.IAS,
    direction = Adjustment.INCREMENT,
    urgencylvl = Urgency.LOW,
    target_by_phase = {
        FlightStage.CLIMB: 180.0,
        FlightStage.CRUISE: 180.0,
        FlightStage.DESCEND: 180.0,
    },
)

max_aob_proc = Procedures(
    name = "Max AoB",
    field = Target_Field.AOB,
    direction = Adjustment.DECREMENT,
    urgencylvl = Urgency.LOW,
    # no target_by_phase — follow rule envelope, no fixed target
)
#CRUISE ONLY max vertical speed
max_vs_proc = Procedures(
    name = "Max VS",
    field = Target_Field.VS,
    direction = Adjustment.DECREMENT,
    urgencylvl = Urgency.MID,
    target_by_phase = {
        FlightStage.CLIMB: 0.0
    },
)
#min vertical speed
min_vs_proc = Procedures(
    name = "Min VS",
    field = Target_Field.VS,
    direction = Adjustment.INCREMENT,
    urgencylvl = Urgency.MID,
    target_by_phase = {
        FlightStage.CLIMB: 0.0
    },
)
vio_procedures: list[Procedures] = [
    min_alt_proc,
    max_alt_proc,
    max_aoa_proc,
    min_aoa_proc,
    max_pos_g_proc,
    max_neg_g_proc,
    max_speed_proc,
    stall_speed_proc,
    max_aob_proc,
    max_vs_proc,
    min_vs_proc
]

def find_procedures(rule_name: str) -> Procedures | None:
    for proc in vio_procedures:
        if proc.name == rule_name:
            return proc
    return None

def format_procedures(proc: Procedures, phase: FlightStage, severity: Severity) -> str:
    if phase in proc.target_by_phase:
        target_text = str(proc.target_by_phase[phase])
    else:
        target_text = "safe attitude"
    return (
        f"RECOVERY PROCEDURE - [{proc.urgencylvl} URGENCY] {proc.direction.name} {proc.field.name} " 
        f"toward target of {target_text} (normal range). Immediate corrective action required.\n"
    )
