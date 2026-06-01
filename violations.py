from dataclasses import dataclass
from flight_frame import FlightFrame
from natops import NatopsRule, FlightStage, Severity, Comparison
from collections import defaultdict
from procedures import find_procedures, format_procedures

@dataclass
class violations:
    frame: FlightFrame
    rule: NatopsRule
    phase: FlightStage
    off_value: float
    threshold: float
    per_vio_severity: Severity

    def __str__(self) -> str:
        return (
            f"Violation occurred at t = {self.frame.time_sec}s. "
            f"Rule that is broken is: {self.rule.name}. "
            f"During the phase of: {self.phase.name}. "
            f"The value that triggers is: {self.off_value}. "
            f"The value it should be is: {self.threshold}. "
            f"Severity level is: {self.per_vio_severity.name}"
        )
    
    

def print_report(all_violations: list[violations], frame_count: float, src_name: str, out_file) -> None:
    if len(all_violations) == 0:
        out_file.write("No Violations Detected \n")
        return
    out_file.write(f"================= Flight Violation Report ================\n")
    out_file.write(f"Source file: {src_name}\n")
    out_file.write(f"Total frame Count: {frame_count }\n")
    out_file.write(f"Total violations are: {len(all_violations)}\n")
    out_file.write("\n")
        
    #counting how many times each rule name appears i.e. rule1: 100 violations
    rule_appear:defaultdict[str, int] = defaultdict(int)
    for v in all_violations:
        rule_appear[v.rule.name] += 1
    
    #print out the violations from the loop above
    for key, value in rule_appear.items():
        out_file.write(f" {key}: {value} violations \n")

    for v in all_violations:
        out_file.write(str(v) + "\n")

    #print out the violations name and total count

def summarize_report(all_violations: list[violations], src_name: str, out_file) -> None:
    if len(all_violations) == 0:
        out_file.write("No Violations Detected\n")
        return
    out_file.write(f"================= Flight Violation Report ================\n")
    out_file.write(f"Source file: {src_name}\n")
    out_file.write(f"Total violations are: {len(all_violations)}\n")
    out_file.write("\n")

    #prioritize rule.name, then time, name and time gap must match to compress into an episode
    sorted_violations = sorted(all_violations, key = lambda v: (v.rule.name, v.frame.time_sec))

    #counting how many times each rule name appears i.e. rule1: 100 violations
    rule_appear:defaultdict[str, int] = defaultdict(int)
    for v in all_violations:
        rule_appear[v.rule.name] += 1
    
    #print out the violations from the loop above
    for key, value in rule_appear.items():
        out_file.write(f" {key}: {value} violations \n")
    
    episode: int = 0
    begin_sec = sorted_violations[0].frame.time_sec     #set initial time
    peak = sorted_violations[0].off_value               #set peak value
    peak_severity = sorted_violations[0].per_vio_severity

    for i in range(1, len(sorted_violations)):
        if sorted_violations[i-1].rule.name != sorted_violations[i].rule.name or sorted_violations[i-1].phase != sorted_violations[i].phase or sorted_violations[i].frame.time_sec - sorted_violations[i-1].frame.time_sec > 0.5:
            end_sec = sorted_violations[i-1].frame.time_sec 
            proc = find_procedures(sorted_violations[i-1].rule.name)
            out_file.write(f"Between {begin_sec} s - {end_sec} s, rule of {sorted_violations[i-1].rule.name} was violated during {sorted_violations[i-1].phase.name} phase, peaked at: {peak}, severity level is: {peak_severity.name}\n")
            if proc is not None:
                out_file.write(format_procedures(proc, sorted_violations[i-1].phase, peak_severity))
                out_file.write(f"\n")
            else:
                out_file.write("No immediate procedures required, follow safety protocal to adjust flight attitude")
                out_file.write(f"\n")
            begin_sec = sorted_violations[i].frame.time_sec
            peak = sorted_violations[i].off_value
            peak_severity = sorted_violations[i].per_vio_severity
            episode += 1
        else:   #need to check comparison to make sure record correct peak, as the peak could be negative
            if sorted_violations[i-1].rule.comparison == Comparison.GREATER_THAN:
                peak = max(peak, sorted_violations[i].off_value)
                peak_severity = max(peak_severity, sorted_violations[i].per_vio_severity)
            else: #comparison.LESS_THAN
                peak = min(peak, sorted_violations[i].off_value)
                peak_severity = max(peak_severity, sorted_violations[i].per_vio_severity)
    #manually write out the last violation episode and procedure
    end_sec = sorted_violations[-2].frame.time_sec
    out_file.write(f"Between {begin_sec} s - {end_sec} s, rule of {sorted_violations[-2].rule.name} was violated during {sorted_violations[-2].phase.name} phase, peaked at: {peak}, severity level is: {peak_severity.name} \n")
    begin_sec = sorted_violations[-1].frame.time_sec
    peak = sorted_violations[-1].off_value
    peak_severity = sorted_violations[-1].per_vio_severity
    episode += 1
    #procedure
    proc = find_procedures(sorted_violations[-1].rule.name)
    if proc is not None:
        out_file.write(format_procedures(proc, sorted_violations[-1].phase, peak_severity))
        out_file.write(f"\n")
    else:
        out_file.write("No immediate procedures required, follow safety protocal to adjust flight attitude")
        out_file.write(f"\n")


    out_file.write(f"Total violations are: {len(all_violations)} frames across {episode} episode\n")

'''
 prev = sorted_violations[i-1]
        curr = sorted_violations[i]
'''