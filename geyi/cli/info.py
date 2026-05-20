"""`geyi info` command."""

from __future__ import annotations

import json

from geyi.analysis import analyze


def run(args) -> int:
    result = analyze(
        args.source,
        spec=args.spec,
        session_root=args.session_root,
        write_session=not args.no_session,
    )
    if args.json:
        print(json.dumps(result.contract.to_dict(), indent=2, sort_keys=True))
    else:
        print_text_report(result)
    return 0


def print_text_report(result) -> None:
    contract = result.contract
    report = result.confidence_report
    intent = contract.intents[0] if contract.intents else None
    intent_name = "%s.%s" % (intent.kind, intent.subkind) if intent else "unknown"

    print("Kernel: %s" % contract.entry)
    print("Intent: %s" % intent_name)
    print("Confidence: %.4f (%s)" % (contract.confidence, contract.confidence_band))
    print("Recommended path: %s" % contract.recommended_path)
    print("Session: %s" % (result.session.path if result.session else "disabled"))
    print("")
    print("Evidence:")
    for item in contract.evidence:
        print("  + %-16s %.2f  %s" % (item.kind, item.confidence, item.claim))
    print("")
    print("Caps:")
    for cap in report.caps:
        print("  - %s caps at %.2f: %s" % (cap.id, cap.cap, cap.reason))
    print("")
    print("Assumptions:")
    if contract.assumptions:
        for item in contract.assumptions:
            print("  - %s" % item.text)
    else:
        print("  none")
    print("")
    print("Unknowns:")
    if contract.unknowns:
        for item in contract.unknowns:
            print("  - %s: %s" % (item.id, item.text))
    else:
        print("  none")
    if contract.rejections:
        print("")
        print("Rejections:")
        for item in contract.rejections:
            print("  - %s hard=%s: %s" % (item.feature, item.hard, item.reason))
    print("")
    print("Required verification: %s" % contract.verification_required)

