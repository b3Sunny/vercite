"""
Main entry point for the claim processing application.

The application processes scientific claims from academic papers using RAG
(Retrieval Augmented Generation) and evaluates the results.
"""

import argparse
from pathlib import Path
from config.env_loader import load_env
from src.services.testcase_service import get_testcase_service
import asyncio

async def main():
    parser = argparse.ArgumentParser(description='Process claims from a test case')
    parser.add_argument('testcase', help='Name of the test case to process')
    parser.add_argument('--create', help='Create a new testcase with given arXiv ID')
    parser.add_argument('--extract', action='store_true', help='Extract claims and references from the testcase PDF')
    parser.add_argument('--download-papers', action='store_true', help='Download referenced arXiv papers')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess claims to generate search queries')
    parser.add_argument('--process', action='store_true', help='Process claims using RAG')
    parser.add_argument('--evaluate', help='Evaluate processing results with specified results directory')
    parser.add_argument('--run', nargs='?', const='', metavar='ARXIV_ID', 
                       help='Run basic testcase workflow. Optionally provide arXiv ID for new testcase.')
    
    args = parser.parse_args()
     
    # Create paths once at the start
    testcase_service = get_testcase_service(args.testcase)

    if args.run or args.preprocess or args.process or args.evaluate:
        load_env()

    if args.run:
        arxiv_id = args.run if args.run else None
        if not await testcase_service.run(arxiv_id):
            return
        return

    if args.create:
        if not testcase_service.create_testcase(args.create):
            return
        
    if args.extract:    
        if not testcase_service.extract():
            return
        return
       
    if args.download_papers:
        if not testcase_service.download_references():
            return
        return
       
    if args.preprocess:
        if not await testcase_service.preprocess_claims():
            return
        return

    if args.process:
        if not await testcase_service.process_claims():
            return
        return
       
    if args.evaluate:
        testcase_service.evaluate_results(Path(args.evaluate))
        return
    
    # If no flags specified, show help
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())