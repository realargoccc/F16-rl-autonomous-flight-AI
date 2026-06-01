from flight_frame import FlightFrame, parse_csv_line
from violations import print_report, summarize_report
from violations import violations
from natops import detect_phase, NatopsRule, is_violated, get_value, compute_severity, f18_rules

def main():
    print("Begin testing")
    csv_path = "/Users/y/Desktop/jsbsim-rl/eval_best.csv"
    stats_collect = []
    rejected = 0

    with open(csv_path) as file:
        next(file)      #skip header
        for line in file:
            result = parse_csv_line(line.strip())
            if result is not None:
                stats_collect.append(result)
            else:
                rejected += 1
    
    all_violations: list[violations] = []
    for frame in stats_collect:
        phase = detect_phase(frame)
        for rule in f18_rules:
            if(is_violated(rule, frame, phase)):
                off_value = get_value(frame, rule.field)
                sev = compute_severity(rule, off_value)
                all_violations.append(violations(frame, rule, phase, off_value, rule.threshold, sev))

    #o_file_loc = "/Users/y/Desktop/jsbsim-rl/sumreport.txt"
    #with open(o_file_loc, "w") as o_file:
        #print_report(all_violations, len(stats_collect), csv_path, o_file)

    sum_o_file = "/Users/y/Desktop/jsbsim-rl/sumreport.txt"
    with open(sum_o_file, "w") as o_file:
        summarize_report(all_violations, csv_path, o_file)
if __name__ == "__main__":
    main()

