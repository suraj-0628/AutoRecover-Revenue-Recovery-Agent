# Razorpay Error Code Resolution Guide

## Error Code: BAD_REQUEST_ERROR (400)
- **Description**: The request is malformed or missing required fields.
- **Resolution**: Verify all mandatory parameters (amount, currency, receipt) are present and correctly typed. Amount must be in paise (integer). Currency must be uppercase ISO 4217.
- **Recovery Protocol**: RETRY_PAYMENT after correcting the request payload. Do NOT escalate to customer — this is a merchant-side fix.

## Error Code: CARD_EXPIRED (54)
- **Description**: The card on file has expired. Expiry date is in the past.
- **Resolution**: Customer must update card details. Silent retries on expired cards WILL fail again.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD — switch to UPI, Netbanking, or a different card. Send notification with link to update payment method. Do NOT schedule_payday_retry on expired card.

## Error Code: INSUFFICIENT_FUNDS (51)
- **Description**: Customer's account has insufficient balance for the transaction.
- **Resolution**: Time retry to payday cycle. Check customer's salary window before retrying.
- **Recovery Protocol**: WAIT_AND_RETRY — schedule retry at next payday (12:01 AM on salary date). Use calculate_payday_window to determine optimal retry time. Maximum 3 silent retries before transitioning to ACTIVE tier.

## Error Code: DO_NOT_HONOR (58)
- **Description**: Bank declined the transaction without specific reason. Common with new cards, international transactions, or bank-side fraud blocks.
- **Resolution**: Check if bank is in degraded health status. Try alternative payment method.
- **Recovery Protocol**: CHECK_BANK_HEALTH first. If bank healthy, try UPDATE_PAYMENT_METHOD to different rail (UPI or Netbanking). If bank degraded, WAIT_AND_RETRY after bank recovery.

## Error Code: GENERIC_DECLINE (96)
- **Description**: Generic decline from issuing bank. No specific reason provided.
- **Resolution**: Similar to DO_NOT_HONOR. Check bank health, try alternative method.
- **Recovery Protocol**: CHECK_BANK_HEALTH → UPDATE_PAYMENT_METHOD if bank healthy → WAIT_AND_RETRY if degraded.

## Error Code: NETWORK_TIMEOUT (127)
- **Description**: Gateway timeout, connection drop, HTTP 5xx error, or 3DS OTP timeout.
- **Resolution**: Transient issue. Customer's bank may be temporarily unreachable.
- **Recovery Protocol**: RETRY_PAYMENT after 5-minute cooldown. Do NOT contact customer for transient timeouts — they are not at fault. Maximum 2 automatic retries before escalating.

## Error Code: RISK_CHECK_FAILED (62)
- **Description**: Razorpay's risk engine flagged the transaction as potentially fraudulent.
- **Resolution**: Requires manual review. Do NOT auto-retry risk-blocked transactions.
- **Recovery Protocol**: ESCALATE_TO_HUMAN immediately. Include risk_score, device_fingerprint, and IP velocity in escalation notes.

## Error Code: MANDATE_INACTIVE (89)
- **Description**: UPI autopay mandate is inactive, expired, or was cancelled by customer.
- **Resolution**: Customer must re-authorize the mandate through their UPI app.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD — send notification with mandate re-authorization link. Do NOT retry mandate without re-authorization. RBI mandates require explicit customer consent for reactivation.

## Error Code: PAYLATER_OTP_EXPIRED (PSP-specific)
- **Description**: LazyPay PayLater OTP verification failed or timed out during authorization.
- **Resolution**: OTP expired before customer could enter it. Common with LazyPay when network is slow.
- **Recovery Protocol**: RETRY_PAYMENT with fresh OTP. If OTP fails 3 times, UPDATE_PAYMENT_METHOD to different rail. Do NOT escalate — this is a transient PSP issue.

## Error Code: PAYLATER_AUTHORIZATION_FAILED (PSP-specific)
- **Description**: LazyPay PayLater authorization declined by PSP after OTP verification.
- **Resolution**: Customer's PayLater credit limit may be exhausted or account suspended.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD — switch to UPI or Card. Do NOT retry PayLater if authorization was explicitly declined.

## Error Code: CARD_INVALID (41)
- **Description**: Card number is invalid or card is not enrolled for online transactions.
- **Resolution**: Customer must use a different card. This is a hard decline.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD immediately. Do NOT retry — retrying invalid cards incurs Visa/Mastercard penalty fines ($0.10/attempt).

## Error Code: EXPIRY_DATE_INVALID (54)
- **Description**: Card expiry date is invalid or in the past.
- **Resolution**: Similar to CARD_EXPIRED. Customer must update card.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD — hard decline, no retries.

## Error Code: CARD_LIMIT_EXCEEDED (61)
- **Description**: Transaction amount exceeds card's per-transaction or daily limit.
- **Resolution**: Customer must use a different card or contact bank to increase limit.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD to different rail. If amount is splittable, consider splitting into smaller transactions.

## Instrument Switch Protocol (CRITICAL)
When description contains "use another payment instrument", "use another payment method", "try another method", "expired", or "invalid card":
- **The current payment method is BROKEN**
- **Silent retries on the same instrument are FORBIDDEN**
- **Immediate action required**: UPDATE_PAYMENT_METHOD to UPI, Netbanking, or Wallet
- **Do NOT**: schedule_payday_retry on broken instrument, RETRY_PAYMENT on same method
- **Do**: Generate smart recovery link with multiple rail options
