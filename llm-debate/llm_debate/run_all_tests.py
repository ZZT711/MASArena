#!/usr/bin/env python3
"""
Summary test script - Run tests for AIME and DROP datasets and generate a statistical report
"""

import subprocess
import os
import json
import time
from datetime import datetime
import numpy as np

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        print(f"Running: {command}")
        # On Windows, try using system encoding first
        import locale
        system_encoding = locale.getpreferredencoding()
        
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            encoding=system_encoding,
            errors='replace'  # Handle encoding errors
        )
        
        if result.returncode == 0:
            print("✓ Command executed successfully")
            stdout = result.stdout if result.stdout else ""
            return True, stdout
        else:
            print(f"✗ Command failed: {result.stderr}")
            stderr = result.stderr if result.stderr else "Unknown error"
            return False, stderr
    except Exception as e:
        print(f"✗ Command exception: {str(e)}")
        return False, str(e)


def parse_evaluation_results(output_text, dataset_name):
    """Parse results from evaluation output"""
    if not output_text or output_text.strip() == "":
        print(f"Warning: {dataset_name} output is empty")
        return {}
    
    lines = output_text.strip().split('\n')
    results = {}
    
    try:
        for line in lines:
            line = line.strip()
            if "Total Questions:" in line:
                # Extract the number after the colon
                value_part = line.split(':')[1].strip().split()[0]
                results['total_questions'] = int(value_part)
            elif "Correct Answers:" in line or "Correct Exact Matches:" in line:
                value_part = line.split(':')[1].strip().split()[0]
                results['correct_count'] = int(value_part)
            elif "Accuracy:" in line and "Average" not in line:
                value_part = line.split(':')[1].strip().split()[0]
                results['accuracy'] = float(value_part)
            elif "Exact Match Rate:" in line and "Average" not in line:
                value_part = line.split(':')[1].strip().split()[0]
                results['exact_match'] = float(value_part)
            elif "Average F1 Score:" in line:
                value_part = line.split(':')[1].strip().split()[0]
                results['f1_score'] = float(value_part)
    except (ValueError, IndexError) as e:
        print(f"Warning: Error parsing {dataset_name} results: {e}")
        print(f"Problematic line: {line}")
    
    return results


def generate_summary_report(aime_results, drop_results):
    """Generate summary report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Safely format values to avoid formatting errors
    def safe_format(value, format_str="{:.4f}", default="N/A"):
        if value is None or value == 'N/A':
            return default
        try:
            if isinstance(value, (int, float)):
                return format_str.format(value)
            return str(value)
        except:
            return default
    
    aime_accuracy = safe_format(aime_results.get('accuracy'))
    drop_exact_match = safe_format(drop_results.get('exact_match'))
    drop_f1_score = safe_format(drop_results.get('f1_score'))
    
    report = f"""
{'='*80}
    Multi-Agent Debate Test Results Summary Report
{'='*80}
Generated at: {timestamp}

{'AIME Math Competition Test Results':-^60}
Dataset: AIME (American Invitational Mathematics Examination)
Number of test questions: {aime_results.get('total_questions', 'N/A')}
Number of correct answers: {aime_results.get('correct_count', 'N/A')}
Accuracy: {aime_accuracy}

{'DROP Reading Comprehension Test Results':-^60}
Dataset: DROP (Discrete Reasoning Over Paragraphs)
Number of test questions: {drop_results.get('total_questions', 'N/A')}
Number of exact match correct answers: {drop_results.get('correct_count', 'N/A')}
Exact match rate: {drop_exact_match}
F1 score: {drop_f1_score}

{'Overall Statistics':-^60}
Total number of test questions: {aime_results.get('total_questions', 0) + drop_results.get('total_questions', 0)}
AIME correct ratio: {aime_results.get('correct_count', 0)}/{aime_results.get('total_questions', 0)}
DROP correct ratio: {drop_results.get('correct_count', 0)}/{drop_results.get('total_questions', 0)}

{'Test Configuration':-^60}
Number of agents: 3
Number of debate rounds: 2
Model: gpt-4o-mini
AIME batch size: 3
DROP batch size: 5

{'Conclusion':-^60}
1. AIME test accuracy: {safe_format(aime_results.get('accuracy', 0), "{:.2%}", "N/A")}
2. DROP test exact match rate: {safe_format(drop_results.get('exact_match', 0), "{:.2%}", "N/A")}
3. DROP test F1 score: {safe_format(drop_results.get('f1_score', 0), "{:.4f}", "N/A")}

{'='*80}
"""
    
    return report


def main():
    """Main function"""
    print("Starting multi-agent debate tests...")
    start_time = time.time()
    
    # Ensure in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    aime_results = {}
    drop_results = {}
    
    # Step 1: Run AIME generation
    print("\n" + "="*50)
    print("Step 1/4: Generating AIME test data...")
    aime_gen_success, output = run_command("python aime/gen_aime.py", cwd=script_dir)
    if not aime_gen_success:
        print("AIME generation failed, skipping AIME evaluation")
        print(f"Error details: {output}")
    else:
        print("AIME generation completed")
        
        # Step 2: Run AIME evaluation
        print("\n" + "="*50)
        print("Step 2/4: Evaluating AIME test results...")
        aime_eval_success, output = run_command("python aime/eval_aime.py", cwd=script_dir)
        if aime_eval_success and output:
            aime_results = parse_evaluation_results(output, "AIME")
            print("AIME evaluation completed")
            print(f"AIME results preview: {aime_results}")
        else:
            print("AIME evaluation failed")
            if output:
                print(f"Error details: {output}")
    
    # Step 3: Run DROP generation
    print("\n" + "="*50)
    print("Step 3/4: Generating DROP test data...")
    drop_gen_success, output = run_command("python drop/gen_drop.py", cwd=script_dir)
    if not drop_gen_success:
        print("DROP generation failed, skipping DROP evaluation")
        print(f"Error details: {output}")
    else:
        print("DROP generation completed")
        
        # Step 4: Run DROP evaluation
        print("\n" + "="*50)
        print("Step 4/4: Evaluating DROP test results...")
        drop_eval_success, output = run_command("python drop/eval_drop.py", cwd=script_dir)
        if drop_eval_success and output:
            drop_results = parse_evaluation_results(output, "DROP")
            print("DROP evaluation completed")
            print(f"DROP results preview: {drop_results}")
        else:
            print("DROP evaluation failed")
            if output:
                print(f"Error details: {output}")
    
    # Generate summary report
    print("\n" + "="*50)
    print("Generating summary report...")
    
    report = generate_summary_report(aime_results, drop_results)
    
    # Save report to file
    report_filename = f"test_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Display report
    print(report)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Detailed report saved to: {report_filename}")
    
    # Save results to JSON
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'aime_results': aime_results,
        'drop_results': drop_results,
        'total_time': end_time - start_time
    }
    
    json_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results data saved to: {json_filename}")


if __name__ == "__main__":
    main() 