"""Faithfulness and groundedness evaluation runner using claim verification and LLM-as-a-judge.

Measures the proportion of factual statements in generated answers that are strictly supported
by reference knowledge base chunks. Contains 40 labeled benchmark cases (20 grounded, 20 planted hallucinations)
and computes Precision, Recall, F1, and Wilson score 95% confidence intervals.
"""

import asyncio
import json
import math
from typing import Any

from agenticrag.config import get_settings
from agenticrag.nodes import check_claim_support
from agenticrag.schemas import ClaimSupportCheck

# 40 Curated Benchmark Evaluation Cases (20 Grounded, 20 Planted Hallucinations)
FAITHFULNESS_BENCHMARK_CASES: list[dict[str, Any]] = [
    # 20 Grounded Supported Cases (ground_truth_supported = True)
    {
        "id": "supp-01",
        "question": "What is the return window?",
        "context": "Items purchased online can be returned within 30 days of delivery with original receipt.",
        "response": "You can return items purchased online within 30 days of delivery, provided you have your original receipt.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-02",
        "question": "How long do refunds take to process?",
        "context": "Refunds are processed to the original payment method within 3 to 5 business days after inspection.",
        "response": "Once inspected, your refund will take between 3 and 5 business days to post to your original payment method.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-03",
        "question": "Is free shipping offered?",
        "context": "Standard ground shipping is free for all domestic orders totaling $50 or more before taxes.",
        "response": "Yes, we offer free standard ground shipping on domestic orders of $50 or more before taxes.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-04",
        "question": "What is the warranty coverage duration?",
        "context": "All electronic accessories include a 1-year limited manufacturer warranty against factory defects.",
        "response": "Our electronic accessories come with a 1-year limited manufacturer warranty covering factory defects.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-05",
        "question": "Can I cancel my order?",
        "context": "Orders can be canceled within 1 hour of placement from your account orders dashboard.",
        "response": "You have up to 1 hour after placing your order to cancel it directly from your account dashboard.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-06",
        "question": "Do you ship to P.O. boxes?",
        "context": "Deliveries to P.O. boxes are supported via USPS Priority Mail only.",
        "response": "Yes, we ship to P.O. boxes using USPS Priority Mail.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-07",
        "question": "How do I reset my password?",
        "context": "Click 'Forgot Password' on the login screen to receive a secure reset link valid for 15 minutes.",
        "response": "To reset your password, click 'Forgot Password' on the sign-in page to get an email link valid for 15 minutes.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-08",
        "question": "What payment methods are accepted?",
        "context": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
        "response": "We currently accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-09",
        "question": "How can I update my shipping address?",
        "context": "Address corrections can only be made while the order is in 'Pending' status by contacting support.",
        "response": "You can update your shipping address by contacting support as long as the order is still in Pending status.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-10",
        "question": "What should I do if my package arrived damaged?",
        "context": "Report damaged deliveries within 48 hours of receipt and provide clear photos of the packaging.",
        "response": "Please report the damage within 48 hours of delivery and share photos of the damaged package.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-11",
        "question": "Are gift cards refundable?",
        "context": "Digital and physical gift cards are non-refundable and cannot be exchanged for cash unless required by law.",
        "response": "Gift cards cannot be refunded or redeemed for cash, except where mandated by local law.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-12",
        "question": "What is the price matching policy?",
        "context": "Price matching applies only to identical in-stock items sold by authorized retailers within 14 days of purchase.",
        "response": "We price match identical, in-stock products from authorized retailers within 14 days of your purchase date.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-13",
        "question": "How do I download my receipt?",
        "context": "Order receipts and VAT invoices can be downloaded as PDF files from the Order History page.",
        "response": "You can download your PDF receipt anytime under the Order History section of your account.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-14",
        "question": "What are your customer support hours?",
        "context": "Live Chat support is active Monday through Friday from 8:00 AM to 8:00 PM Eastern Time.",
        "response": "Live Chat is available weekdays between 8:00 AM and 8:00 PM EST.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-15",
        "question": "Can I exchange an item for a different size?",
        "context": "Direct size exchanges are free of charge within 30 days if the replacement size is in stock.",
        "response": "Yes, direct size exchanges are complimentary within 30 days, subject to inventory availability.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-16",
        "question": "Do you charge restocking fees?",
        "context": "Opened electronics returned without original packaging are subject to a 10% restocking fee.",
        "response": "A 10% restocking fee applies only if opened electronics are returned without their original packaging.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-17",
        "question": "How do I track my delivery?",
        "context": "Tracking numbers are generated and emailed automatically as soon as carrier dispatch occurs.",
        "response": "You will receive an automated email containing your tracking number once your package has shipped.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-18",
        "question": "Is international shipping offered to Germany?",
        "context": "International shipping is offered to European Union nations via DHL Express within 4-7 business days.",
        "response": "Yes, we ship to Germany and other EU countries using DHL Express in approximately 4 to 7 business days.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-19",
        "question": "Can I use multiple promo codes on one purchase?",
        "context": "Only one promotional code may be applied per checkout transaction.",
        "response": "Only a single promotional discount code can be used per order.",
        "ground_truth_supported": True,
    },
    {
        "id": "supp-20",
        "question": "What happens if my package is lost?",
        "context": "If tracking shows no movement for 7 consecutive days, our claims team will initiate a carrier investigation.",
        "response": "Our team opens a carrier investigation if the shipment tracking shows no updates for 7 consecutive days.",
        "ground_truth_supported": True,
    },
    # 20 Planted Hallucination / Unsupported Cases (ground_truth_supported = False)
    {
        "id": "halluc-01",
        "question": "What is the return window?",
        "context": "Items purchased online can be returned within 30 days of delivery with original receipt.",
        "response": "You have 90 days to return any item, no questions asked, and we provide cash refunds on the spot.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-02",
        "question": "What is the customer support phone number?",
        "context": "Support is available exclusively through our web Live Chat and online ticketing portal.",
        "response": "You can call our dedicated support hotline at 1-800-555-0199 24/7.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-03",
        "question": "What does the warranty cover?",
        "context": "The 1-year limited warranty covers defects in materials and workmanship under normal use.",
        "response": "The warranty includes unlimited accidental damage, flood protection, and lifetime free battery replacement.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-04",
        "question": "Is free shipping offered?",
        "context": "Standard ground shipping is free for all domestic orders totaling $50 or more before taxes.",
        "response": "Every order qualifies for free overnight FedEx air shipping with no minimum spend required.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-05",
        "question": "Can I cancel my order after 2 hours?",
        "context": "Orders can be canceled within 1 hour of placement from your account orders dashboard.",
        "response": "Yes, orders can be canceled anytime up until the delivery truck arrives at your door.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-06",
        "question": "Do you accept cryptocurrency?",
        "context": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
        "response": "We gladly accept Bitcoin, Ethereum, and Dogecoin for all online purchases.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-07",
        "question": "What is the restocking fee for clothing?",
        "context": "Clothing items in original condition with tags attached can be returned with zero restocking fee.",
        "response": "There is a mandatory 25% cleaning and restocking fee deducted from all clothing returns.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-08",
        "question": "How long is password reset link valid?",
        "context": "Click 'Forgot Password' on the login screen to receive a secure reset link valid for 15 minutes.",
        "response": "The password reset link remains active and valid for 30 consecutive days.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-09",
        "question": "Do you offer military discounts?",
        "context": "We currently do not offer specialized student or military discounts at this time.",
        "response": "Yes, active military personnel receive an automatic 30% discount on every purchase upon presenting military ID.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-10",
        "question": "Are shipping boxes recyclable?",
        "context": "Standard ground delivery arrives within 3-5 business days.",
        "response": "All of our shipping boxes are made from 100% biodegradable seaweed packaging that dissolves in water.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-11",
        "question": "What is your price match guarantee timeframe?",
        "context": "Price matching applies only to identical in-stock items within 14 days of purchase.",
        "response": "We will match any competitor price for up to 365 days after your purchase date.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-12",
        "question": "Can I pick up my order in a store?",
        "context": "We operate exclusively as a direct-to-consumer online business without brick-and-mortar retail storefronts.",
        "response": "You can select in-store curbside pickup at any of our 500 retail stores nationwide.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-13",
        "question": "What is the policy for items lost in transit?",
        "context": "If tracking shows no movement for 7 consecutive days, our claims team will initiate a carrier investigation.",
        "response": "If lost, we immediately wire $1000 cash compensation to your bank account with no claim form required.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-14",
        "question": "How do direct size exchanges work?",
        "context": "Direct size exchanges are free of charge within 30 days if the replacement size is in stock.",
        "response": "Size exchanges require paying an administrative surcharge of $15 plus international shipping.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-15",
        "question": "Are gift cards transferable?",
        "context": "Digital and physical gift cards are non-refundable and cannot be exchanged for cash.",
        "response": "Gift cards can be converted to physical gold coins upon request to our treasury department.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-16",
        "question": "What are support hours on Sunday?",
        "context": "Live Chat support is active Monday through Friday from 8:00 AM to 8:00 PM Eastern Time.",
        "response": "Our live support agents are staffed around the clock 24/7/365 including Sundays.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-17",
        "question": "Do refurbished products have warranty?",
        "context": "Refurbished items are covered under a 90-day certified reconditioned warranty.",
        "response": "Refurbished items have no warranty whatsoever and are sold completely as-is with all sales final.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-18",
        "question": "Can I combine two orders into one box?",
        "context": "Orders cannot be combined once placed due to our automated warehouse fulfillment workflows.",
        "response": "Yes, just click 'Combine Orders' in your cart to merge any orders placed in the same week.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-19",
        "question": "Can I return an item without receipt?",
        "context": "A valid order number or original sales receipt is mandatory for all returns and exchanges.",
        "response": "No receipt or order number is needed; simply mail the item back with your name written on a note.",
        "ground_truth_supported": False,
    },
    {
        "id": "halluc-20",
        "question": "How many promo codes can I apply?",
        "context": "Only one promotional code may be applied per checkout transaction.",
        "response": "You can stack up to 10 promo codes simultaneously to maximize your discount.",
        "ground_truth_supported": False,
    },
]


def wilson_score_interval(p_hat: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 1.645
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * (n**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


async def run_faithfulness_benchmark(
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Runs claim-support evaluator against the 40-case labeled benchmark and outputs precision, recall, and 95% CI."""
    eval_cases = cases or FAITHFULNESS_BENCHMARK_CASES
    settings = get_settings()

    if not settings.groq_api_key:
        print("[Faithfulness Eval] GROQ_API_KEY is not set. Marking metric as UNMEASURED.")
        return {
            "status": "UNMEASURED",
            "reason": "GROQ_API_KEY unavailable",
            "total_cases": len(eval_cases),
        }

    print(f"[Faithfulness Eval] Evaluating {len(eval_cases)} labeled cases...")
    tp = 0  # Predicted supported, actual supported
    tn = 0  # Predicted unsupported, actual unsupported
    fp = 0  # Predicted supported, actual unsupported (hallucination missed)
    fn = 0  # Predicted unsupported, actual supported (false alarm)
    detailed_results = []

    for idx, c in enumerate(eval_cases):
        q = c["question"]
        ctx = c["context"]
        resp = c["response"]
        expected = c["ground_truth_supported"]

        try:
            check: ClaimSupportCheck = await check_claim_support(
                question=q, context=ctx, response=resp
            )
            predicted = check.supported

            if expected and predicted:
                tp += 1
            elif not expected and not predicted:
                tn += 1
            elif not expected and predicted:
                fp += 1
            else:
                fn += 1

            status = "CORRECT" if (predicted == expected) else "INCORRECT"
            detailed_results.append(
                {
                    "id": c.get("id", f"case-{idx}"),
                    "expected": expected,
                    "predicted": predicted,
                    "status": status,
                    "reason": check.reason,
                    "unsupported_claims": check.unsupported_claims,
                }
            )
        except Exception as err:
            print(f"[Faithfulness Eval Error] Case {idx + 1} failed: {err}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    ci_lower, ci_upper = wilson_score_interval(accuracy, total, confidence=0.95)

    summary = {
        "status": "COMPLETED",
        "total_cases": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "ci_95_percent": [ci_lower, ci_upper],
    }

    print("\n================ FAITHFULNESS EVALUATION SCORECARD ================")
    print(f"Total Cases Evaluated: {total} (20 Supported, 20 Hallucinated)")
    print(f"Accuracy:  {accuracy:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print("===================================================================\n")

    return {**summary, "details": detailed_results}


if __name__ == "__main__":
    report = asyncio.run(run_faithfulness_benchmark())
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
