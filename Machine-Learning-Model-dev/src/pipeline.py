"""
RegQuest ML Data Processing Pipeline
Loads, validates, cleans, and standardizes historical registrar datasets.
Exports cleaned dataset, daily time-series dataset, rejected records, and a data-quality audit report.
"""

import os
import glob
import csv
import json
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Directory references
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")

# Canonical Credential Type Mapping based on Registrar Logbook Legend & Backend Codes
CREDENTIAL_MAP = {
    # 1. TOR
    "1": "TOR",
    "TOR": "TOR",
    "TOR-GRANTED": "TOR",
    "TOR GRANTED": "TOR",
    "TOR- GRANTED": "TOR",
    "TRANSCRIPT": "TOR",
    # 2. HD
    "2": "HD",
    "HD": "HD",
    "HONORABLE DISMISSAL": "HD",
    # 3. CAV
    "3": "CAV",
    "CAV": "CAV",
    # 4. Certifications
    "4": "CERTIFICATION",
    "4A": "COE",
    "4a": "COE",
    "OFFICIALLY ENROLLED": "COE",
    "4B": "GPA_CERT",
    "4b": "GPA_CERT",
    "GPA": "GPA_CERT",
    "4C": "GRAD_CERT",
    "4c": "GRAD_CERT",
    "GRADUATED": "GRAD_CERT",
    "4D": "UNITS_CERT",
    "4d": "UNITS_CERT",
    "EARNED UNITS": "UNITS_CERT",
    "4E": "ENROLLED_SUBJ_CERT",
    "4e": "ENROLLED_SUBJ_CERT",
    "SUBJECTS ENROLLED": "ENROLLED_SUBJ_CERT",
    "4F": "CAR",
    "4f": "CAR",
    "CAR": "CAR",
    "CERTIFICATE OF ACADEMIC RECORDS": "CAR",
    "4G": "GRADES_CERT",
    "4g": "GRADES_CERT",
    "SUBJECTS W/ GRADES": "GRADES_CERT",
    "4H": "CONVERSION_CERT",
    "4h": "CONVERSION_CERT",
    "USTP CONVERSION": "CONVERSION_CERT",
    "4I": "ENGLISH_CERT",
    "4i": "ENGLISH_CERT",
    "ENGLISH MEDIUM": "ENGLISH_CERT",
    "4J": "GRADING_SYS_CERT",
    "4j": "GRADING_SYS_CERT",
    "GRADING SYSTEM": "GRADING_SYS_CERT",
    "4K": "HONOR_CERT",
    "4k": "HONOR_CERT",
    "CUM LAUDE": "HONOR_CERT",
    "LATIN HONOR": "HONOR_CERT",
    "4L": "GRADUATING_CERT",
    "4l": "GRADUATING_CERT",
    "GRADUATING": "GRADUATING_CERT",
    # 5. Evaluation
    "5": "EVAL",
    "EVALUATION": "EVAL",
    # 6. Form 137
    "6": "F137",
    "FORM 137": "F137",
    # 7. Diploma Replacement
    "7": "DIPLOMA",
    "DIPLOMA REPLACEMENT": "DIPLOMA",
    "DIPLOMA": "DIPLOMA",
    # 8. WES
    "8": "WES",
    "WES": "WES",
    # 9. Permit to study
    "9": "STUDY_PERMIT",
    "PERMIT TO STUDY": "STUDY_PERMIT",
    # 10. Verification
    "10": "VERIFICATION",
    "VERIFICATION": "VERIFICATION",
    "SCHOOL VERIFICATION": "VERIFICATION",
    # Miscellaneous specific certificates
    "NSTP SERIAL NO": "NSTP_CERT",
    "NSTP SERIAL NO,": "NSTP_CERT",
    "LETTER OF NO OBJECTION": "NO_OBJECTION_CERT"
}

# Purpose standardization
def standardize_purpose(raw_purpose: str) -> str:
    if not raw_purpose:
        return "Unspecified"
    p = raw_purpose.strip().upper()
    if "EVAL" in p:
        return "Evaluation"
    if "EMPLOY" in p or "JOB" in p or "WORK" in p:
        return "Employment"
    if "SCHOLAR" in p or "CHED" in p or "SUBSIDY" in p or "GSIS" in p or "ASSISTANCE" in p:
        return "Scholarship"
    if "BOARD" in p or "PRC" in p or "EXAM" in p:
        return "Board Exam"
    if "TOR" in p and "GRANT" in p:
        return "TOR-Granted"
    if "STUD" in p or "MASTER" in p or "LAW" in p or "ADVANCE" in p:
        return "Advanced Studies"
    if "TRANS" in p:
        return "Transfer"
    if "RANK" in p:
        return "Ranking"
    if "FILE" in p or "COPY" in p or "PERSONAL" in p:
        return "Personal File"
    if "VISA" in p or "ABROAD" in p or "PASSPORT" in p:
        return "Visa / Travel"
    if "PROMOT" in p:
        return "Promotion"
    if "CIVIL SERVICE" in p or "CSC" in p:
        return "Civil Service"
    return "Other"


def parse_raw_credential(raw_cred: str):
    """
    Parses single or comma-separated credential codes.
    Returns: primary_code, all_codes_list, is_multi_doc, doc_count
    """
    if not raw_cred:
        return "UNKNOWN", ["UNKNOWN"], False, 1
    
    parts = [p.strip() for p in re.split(r"[,/]+", raw_cred) if p.strip()]
    if not parts:
        return "UNKNOWN", ["UNKNOWN"], False, 1
    
    canonical_codes = []
    for part in parts:
        clean_part = part.strip()
        matched = CREDENTIAL_MAP.get(clean_part) or CREDENTIAL_MAP.get(clean_part.upper())
        if matched:
            canonical_codes.append(matched)
        else:
            # Fallback search inside dictionary
            found = False
            for k, v in CREDENTIAL_MAP.items():
                if k.upper() == clean_part.upper():
                    canonical_codes.append(v)
                    found = True
                    break
            if not found:
                canonical_codes.append(clean_part.upper())
                
    primary = canonical_codes[0] if canonical_codes else "UNKNOWN"
    return primary, canonical_codes, len(canonical_codes) > 1, len(canonical_codes)


def clean_and_parse_date(date_str: str, file_name: str = "", row_idx: int = -1):
    """
    Deterministic date parsing and typo resolution rule-engine.
    Guarantees zero silent mutations; every correction is audit logged.
    """
    if not date_str:
        return None, "Empty date"
        
    d = date_str.strip()
    
    # Deterministic typo repair rules
    if d == "9/1/20/2025":
        return datetime(2025, 9, 1), "Repetitive separator 9/1/20/2025 -> 2025-09-01"
    if d == "10//14/2025":
        return datetime(2025, 10, 14), "Double slash 10//14/2025 -> 2025-10-14"
    if d == "4//28/2025":
        return datetime(2025, 4, 28), "Double slash 4//28/2025 -> 2025-04-28"
    if d == "429/2025":
        return datetime(2025, 4, 29), "Missing slash 429/2025 -> 2025-04-29"
    if d == "1/4/1900":
        # In July Data row 206, batch requested date is 6/30/2025 for scheduled release 7/7/2025
        return datetime(2025, 6, 30), "Excel integer '4' epoch overflow in 6/30/2025 batch -> 2025-06-30"
    if d == "GSIS":
        return None, "Shifted column text 'GSIS'"
        
    # Missing slash before year (e.g. 5/302025 -> 5/30/2025)
    m_missing_slash = re.match(r"^(\d{1,2})/(\d{1,2})2025$", d)
    if m_missing_slash:
        m, day = map(int, m_missing_slash.groups())
        return datetime(2025, m, day), f"Missing year slash {d} -> 2025-{m:02d}-{day:02d}"
        
    # Standard M/D/YYYY
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", d)
    if m:
        month, day, year = map(int, m.groups())
        # Clerical year correction for 2015 in Dec 2025 logbook or 2925/0205
        if year == 2015 and "December" in file_name:
            year = 2025
            correction_note = f"Typo year 2015 in Dec 2025 batch -> 2025-{month:02d}-{day:02d}"
        elif year in (205, 2925):
            year = 2025
            correction_note = f"Malformed year {year} -> 2025-{month:02d}-{day:02d}"
        else:
            correction_note = "Valid standard date"
            
        try:
            return datetime(year, month, day), correction_note
        except ValueError as e:
            return None, f"Invalid calendar date numbers: {d} ({e})"
            
    # ISO formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(d, fmt), "Parsed via datetime"
        except ValueError:
            pass
            
    return None, f"Unrecognized date format: {d}"


def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculates Monday through Friday operating days between two dates."""
    if start_date > end_date:
        return 0
    cur = start_date
    business_days = 0
    while cur <= end_date:
        if cur.weekday() < 5:  # Mon-Fri
            business_days += 1
        cur += timedelta(days=1)
    return business_days


def run_pipeline():
    """Executes the full extraction, cleaning, and transformation pipeline."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
        
    print(f"=== RegQuest Data Processing Pipeline ===")
    print(f"Found {len(files)} raw monthly files.")
    
    raw_records_count = 0
    cleaned_records = []
    rejected_records = []
    corrections_log = []
    
    file_stats = {}
    
    for f in files:
        fname = os.path.basename(f)
        with open(f, 'r', encoding='utf-8', errors='replace') as fp:
            rows = list(csv.reader(fp))
            
        # Lines 0-6 are headers, Line 7 is column names, Line 8+ is data
        data_rows = rows[8:] if len(rows) > 8 else []
        file_valid = 0
        file_rejected = 0
        
        for r_idx, r in enumerate(data_rows, start=9):
            if not any(c.strip() for c in r):
                continue
            raw_records_count += 1
            
            raw_no = r[0].strip() if len(r) > 0 else ""
            raw_name = r[1].strip() if len(r) > 1 else ""
            raw_course = r[2].strip() if len(r) > 2 else ""
            raw_grad2018 = r[3].strip() if len(r) > 3 else ""
            raw_cred = r[4].strip() if len(r) > 4 else ""
            raw_purpose = r[5].strip() if len(r) > 5 else ""
            raw_req_d = r[6].strip() if len(r) > 6 else ""
            raw_sched_d = r[7].strip() if len(r) > 7 else ""
            raw_proc_t = r[8].strip() if len(r) > 8 else ""
            
            # Special case for column-shifted row 961 in June Data.csv
            if fname == "June Data.csv" and r_idx == 961:
                req_dt = datetime(2025, 6, 19)
                sched_dt = datetime(2025, 6, 26)
                purpose_std = "Employment"
                corrections_log.append({
                    "file": fname,
                    "row": r_idx,
                    "field": "multiple",
                    "raw_value": "Shifted row ('GSIS' in date)",
                    "corrected_value": "Req: 2025-06-19, Sched: 2025-06-26, Purpose: Employment",
                    "reason": "Column shift reconstruction"
                })
            else:
                req_dt, req_note = clean_and_parse_date(raw_req_d, fname, r_idx)
                sched_dt, sched_note = clean_and_parse_date(raw_sched_d, fname, r_idx)
                purpose_std = standardize_purpose(raw_purpose)
                
                if req_note != "Valid standard date" and req_dt:
                    corrections_log.append({
                        "file": fname,
                        "row": r_idx,
                        "field": "Date Requested",
                        "raw_value": raw_req_d,
                        "corrected_value": req_dt.strftime("%Y-%m-%d"),
                        "reason": req_note
                    })
                if sched_note != "Valid standard date" and sched_dt:
                    corrections_log.append({
                        "file": fname,
                        "row": r_idx,
                        "field": "Scheduled Date for Release",
                        "raw_value": raw_sched_d,
                        "corrected_value": sched_dt.strftime("%Y-%m-%d"),
                        "reason": sched_note
                    })

            if not req_dt:
                rejected_records.append({
                    "file": fname,
                    "row": r_idx,
                    "raw_row": r,
                    "reason": f"Unresolvable Date Requested: {raw_req_d}"
                })
                file_rejected += 1
                continue
                
            primary_cred, all_creds, is_multi, num_docs = parse_raw_credential(raw_cred)
            
            # Compute lead times
            if sched_dt:
                lead_cal_days = (sched_dt - req_dt).days
                lead_biz_days = calculate_business_days(req_dt, sched_dt)
            else:
                lead_cal_days = None
                lead_biz_days = None
                
            rec = {
                "source_file": fname,
                "source_row": r_idx,
                "original_no": raw_no,
                "student_identifier": raw_name,
                "course": raw_course,
                "graduated_2018_onwards": bool(raw_grad2018),
                "credential_raw": raw_cred,
                "primary_credential": primary_cred,
                "all_credentials": ";".join(all_creds),
                "is_multi_document": is_multi,
                "num_documents": num_docs,
                "purpose_raw": raw_purpose,
                "purpose_category": purpose_std,
                "date_requested": req_dt.strftime("%Y-%m-%d"),
                "date_requested_weekday": req_dt.strftime("%A"),
                "date_requested_month": req_dt.month,
                "date_requested_year": req_dt.year,
                "scheduled_release_date": sched_dt.strftime("%Y-%m-%d") if sched_dt else "",
                "scheduled_lead_calendar_days": lead_cal_days,
                "scheduled_lead_business_days": lead_biz_days,
                "raw_processing_time_field": raw_proc_t
            }
            cleaned_records.append(rec)
            file_valid += 1
            
        file_stats[fname] = {"raw": len(data_rows), "valid": file_valid, "rejected": file_rejected}

    print(f"Total raw rows: {raw_records_count}")
    print(f"Total cleaned & validated requests: {len(cleaned_records)}")
    print(f"Total rejected rows: {len(rejected_records)}")
    print(f"Total audited corrections applied: {len(corrections_log)}")

    # Sort cleaned records chronologically
    cleaned_records.sort(key=lambda x: (x["date_requested"], x["source_file"], x["source_row"]))

    # Deduplication check
    seen_hashes = Counter()
    duplicates_count = 0
    for r in cleaned_records:
        h = (r["student_identifier"], r["date_requested"], r["primary_credential"], r["purpose_category"])
        seen_hashes[h] += 1
        if seen_hashes[h] > 1:
            duplicates_count += 1

    # Export Cleaned Requests CSV
    cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_requests.csv")
    fieldnames = list(cleaned_records[0].keys())
    with open(cleaned_csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_records)
    print(f"Saved: {cleaned_csv_path}")

    # Export Rejected Records CSV
    rejected_csv_path = os.path.join(PROCESSED_DATA_DIR, "rejected_records.csv")
    with open(rejected_csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=["file", "row", "raw_row", "reason"])
        writer.writeheader()
        writer.writerows(rejected_records)
    print(f"Saved: {rejected_csv_path}")

    # Aggregate Daily Time Series
    daily_aggregates = defaultdict(lambda: {
        "total_requests": 0,
        "tor_count": 0,
        "hd_count": 0,
        "eval_count": 0,
        "car_count": 0,
        "coe_count": 0,
        "other_cert_count": 0,
        "multi_doc_requests": 0,
        "total_documents_requested": 0
    })

    for r in cleaned_records:
        d = r["date_requested"]
        daily_aggregates[d]["total_requests"] += 1
        daily_aggregates[d]["total_documents_requested"] += r["num_documents"]
        if r["is_multi_document"]:
            daily_aggregates[d]["multi_doc_requests"] += 1
            
        cred = r["primary_credential"]
        if cred == "TOR":
            daily_aggregates[d]["tor_count"] += 1
        elif cred == "HD":
            daily_aggregates[d]["hd_count"] += 1
        elif cred == "EVAL":
            daily_aggregates[d]["eval_count"] += 1
        elif cred == "CAR":
            daily_aggregates[d]["car_count"] += 1
        elif cred == "COE":
            daily_aggregates[d]["coe_count"] += 1
        else:
            daily_aggregates[d]["other_cert_count"] += 1

    daily_ts_records = []
    for d_str in sorted(daily_aggregates.keys()):
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        agg = daily_aggregates[d_str]
        daily_ts_records.append({
            "date": d_str,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "day_of_week": dt.weekday(),
            "weekday_name": dt.strftime("%A"),
            "is_weekend": int(dt.weekday() >= 5),
            "is_operating_day": int(dt.weekday() < 5),
            "total_requests": agg["total_requests"],
            "total_documents_requested": agg["total_documents_requested"],
            "tor_count": agg["tor_count"],
            "hd_count": agg["hd_count"],
            "eval_count": agg["eval_count"],
            "car_count": agg["car_count"],
            "coe_count": agg["coe_count"],
            "other_cert_count": agg["other_cert_count"],
            "multi_doc_requests": agg["multi_doc_requests"]
        })

    daily_csv_path = os.path.join(PROCESSED_DATA_DIR, "daily_time_series.csv")
    daily_fields = list(daily_ts_records[0].keys())
    with open(daily_csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=daily_fields)
        writer.writeheader()
        writer.writerows(daily_ts_records)
    print(f"Saved: {daily_csv_path}")

    # Build and Save Data Quality Audit Report
    quality_report = {
        "generation_timestamp": datetime.now().isoformat(),
        "total_source_files": len(files),
        "total_raw_rows": raw_records_count,
        "total_cleaned_records": len(cleaned_records),
        "total_rejected_records": len(rejected_records),
        "total_exact_duplicates": duplicates_count,
        "date_corrections_applied": len(corrections_log),
        "date_coverage": {
            "earliest_date_requested": cleaned_records[0]["date_requested"] if cleaned_records else None,
            "latest_date_requested": cleaned_records[-1]["date_requested"] if cleaned_records else None,
            "distinct_request_dates": len(daily_ts_records)
        },
        "file_breakdown": file_stats,
        "primary_credential_distribution": dict(Counter([r["primary_credential"] for r in cleaned_records]).most_common(20)),
        "purpose_category_distribution": dict(Counter([r["purpose_category"] for r in cleaned_records]).most_common(15)),
        "corrections_sample": corrections_log[:20]
    }

    report_json_path = os.path.join(REPORTS_DIR, "data_quality_report.json")
    with open(report_json_path, 'w', encoding='utf-8') as fp:
        json.dump(quality_report, fp, indent=2)
    print(f"Saved: {report_json_path}")
    print("=== Pipeline Complete ===")
    return quality_report

if __name__ == "__main__":
    run_pipeline()

