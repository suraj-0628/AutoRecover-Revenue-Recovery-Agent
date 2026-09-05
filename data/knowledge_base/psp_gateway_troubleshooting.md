# PSP & Gateway Troubleshooting Guide

## LazyPay PayLater

### Common Failure Modes
1. **OTP_EXPIRED (PSP-LP-001)**: OTP timeout during PayLater authorization
   - **Root cause**: Network latency between Razorpay and LazyPay PSP, or customer delayed OTP entry
   - **Resolution**: RETRY_PAYMENT with fresh OTP. Maximum 2 retries before switching rail.
   - **Don't**: Escalate to human — this is a transient PSP issue

2. **AUTHORIZATION_DECLINED (PSP-LP-002)**: LazyPay declined authorization after OTP
   - **Root cause**: Customer's PayLater credit limit exhausted, account suspended, or merchant not enabled for PayLater
   - **Resolution**: UPDATE_PAYMENT_METHOD — switch to UPI, Card, or Netbanking
   - **Don't**: Retry PayLater if explicitly declined

3. **SETTLEMENT_DELAY (PSP-LP-003)**: Payment captured but settlement pending
   - **Root cause**: LazyPay settlement cycle (T+2 for confirmed, T+7 for disputed)
   - **Resolution**: No action needed — inform merchant of settlement timeline
   - **Don't**: Initiate refund for settlement delays

4. **NETWORK_TIMEOUT (PSP-LP-004)**: Connection timeout to LazyPay gateway
   - **Root cause**: LazyPay gateway under load or maintenance window
   - **Resolution**: WAIT_AND_RETRY after 10 minutes. Check LazyPay status page.
   - **Don't**: Contact customer — this is infrastructure-side

### LazyPay-Specific Recovery Rules
- LazyPay has a 15-minute authorization window — OTP must be entered within this time
- Maximum 3 OTP attempts per transaction
- PayLater amount must be between ₹100 and ₹1,00,000
- LazyPay charges merchant 2% processing fee (factor in for discount calculations)

## NPCI UPI Gateway

### Common Failure Modes
1. **NPCI_Z9_TIMEOUT (PSP-UPI-001)**: NPCI gateway timeout
   - **Root cause**: NPCI processing infrastructure under load (common during salary days, festival seasons)
   - **Resolution**: RETRY_PAYMENT after 5-minute cooldown. Check NPCI status page for outages.
   - **Don't**: Escalate immediately — NPCI timeouts are usually transient

2. **VPA_NOT_FOUND (PSP-UPI-002)**: Customer's UPI VPA could not be resolved
   - **Root cause**: Customer entered invalid VPA, or PSP doesn't support that VPA format
   - **Resolution**: UPDATE_PAYMENT_METHOD — ask customer to verify VPA or use QR code
   - **Don't**: Retry same VPA — it will fail again

3. **BLOCKED_BY_ISSUER (PSP-UPI-003)**: Bank blocked UPI transaction
   - **Root cause**: Bank's fraud system flagged the transaction, or customer's UPI is disabled
   - **Resolution**: CHECK_BANK_HEALTH → if healthy, customer must enable UPI at bank → RETRY
   - **Don't**: Auto-retry — bank block requires customer action

4. **DEBIT_FAILED_CREDIT_SUCCESS (PSP-UPI-004)**: Money debited from customer but not credited to merchant
   - **Root cause**: NPCI settlement failure — rare but serious
   - **Resolution**: ESCALATE_TO_HUMAN immediately. Include NPCI RRN (Retrieval Reference Number) for traceability.
   - **Don't**: Auto-refund — wait for NPCI reconciliation (T+1)

### UPI-Specific Recovery Rules
- UPI transaction limit: ₹1,00,000 per transaction (₹5,00,000 for UPI 2.0)
- UPI collect requests expire after 2 minutes
- UPI autopay mandates require re-authorization if bank changes core banking system
- NPCI downtime usually affects all PSPs simultaneously — check multiple bank health indicators

## HDFC Netbanking

### Common Failure Modes
1. **HTTP_503_SERVICE_UNAVAILABLE (PSP-HDFC-001)**: HDFC gateway returns 503
   - **Root cause**: HDFC maintenance window (typically 2:00 AM - 6:00 AM IST) or infrastructure failure
   - **Resolution**: WAIT_AND_RETRY after maintenance window. Check HDFC status page.
   - **Don't**: Contact customer during maintenance — they can't fix it

2. **SESSION_EXPIRED (PSP-HDFC-002)**: Netbanking session expired during payment
   - **Root cause**: Customer took too long to complete authentication, or browser session timed out
   - **Resolution**: RETRY_PAYMENT — generate fresh payment link. Customer must re-authenticate.
   - **Don't**: Blame customer — session timeouts are common with slow connections

3. **OTP_NOT_RECEIVED (PSP-HDFC-003)**: Customer didn't receive OTP from HDFC
   - **Root cause**: SMS delivery delay, DND (Do Not Disturb) blocking OTP SMS, or HDFC OTP service issue
   - **Resolution**: RETRY_PAYMENT after 5 minutes. If OTP fails 3 times, UPDATE_PAYMENT_METHOD.
   - **Don't**: Escalate for OTP delivery issues — they resolve with retry

4. **TRANSACTION_DECLINED_BY_BANK (PSP-HDFC-004)**: Bank explicitly declined
   - **Root cause**: Transaction limit exceeded, bank security block, or insufficient balance
   - **Resolution**: CHECK_BANK_HEALTH → UPDATE_PAYMENT_METHOD to different bank
   - **Don't**: Retry on same bank — explicit decline means bank won't approve

### HDFC-Specific Recovery Rules
- HDFC has the highest gateway timeout rate among Indian banks (avg 3.2% during peak hours)
- HDFC maintenance window: 2:00 AM - 6:00 AM IST daily
- HDFC OTP delivery is SMS-only (no email fallback for netbanking)
- HDFC UPI transactions are processed through NPCI (not HDFC gateway)

## ICICI Netbanking

### Common Failure Modes
1. **3DS_AUTH_TIMEOUT (PSP-ICICI-001)**: 3D Secure authentication timeout
   - **Root cause**: ICICI's 3DS server slow to respond, or customer delayed OTP entry
   - **Resolution**: RETRY_PAYMENT with fresh 3DS request. Maximum 2 retries.
   - **Don't**: Escalate — ICICI 3DS timeouts are transient

2. **VPN_BLOCK (PSP-ICICI-002)**: ICICI blocks transactions from VPN IPs
   - **Root cause**: Customer using VPN, ICICI fraud system flags VPN IPs
   - **Resolution**: Ask customer to disable VPN and retry. UPDATE_PAYMENT_METHOD if customer can't disable VPN.
   - **Don't**: Auto-retry — VPN block will persist

### ICICI-Specific Recovery Rules
- ICICI has the best UPI success rate (97.8% among major banks)
- ICICI netbanking supports email OTP as backup (unlike HDFC)
- ICICI auto-debit mandates have 48-hour retry window

## SBI Netbanking

### Common Failure Modes
1. **MAINTENANCE_WINDOW (PSP-SBI-001)**: SBI scheduled maintenance
   - **Root cause**: SBI maintenance typically on first Sunday of month, 10:00 AM - 2:00 PM IST
   - **Resolution**: WAIT_AND_RETRY after maintenance. Queue retry for post-maintenance window.
   - **Don't**: Escalate during maintenance — it's scheduled

2. **HIGH_VOLUME_THROTTLE (PSP-SBI-002)**: SBI throttles during high volume (salary day)
   - **Root cause**: SBI infrastructure can't handle peak load (1st-5th of month)
   - **Resolution**: WAIT_AND_RETRY with exponential backoff. Avoid retrying during 10:00 AM - 2:00 PM on salary days.
   - **Don't**: Mass-retry all SBI failures — compound the load

### SBI-Specific Recovery Rules
- SBI has the highest failure rate during salary days (1st-5th of month)
- SBI maintenance is predictable (first Sunday) — schedule retries around it
- SBI netbanking supports both OTP and grid card authentication
- SBI UPI transactions are processed through NPCI

## Card Network 3DS Drops

### Common Failure Modes
1. **VISA_3DS_TIMEOUT (PSP-VISA-001)**: Visa 3D Secure timeout
   - **Root cause**: Visa's Access Control Server (ACS) slow to respond, or cross-border routing issue
   - **Resolution**: RETRY_PAYMENT after 5-minute cooldown. If persistent, UPDATE_PAYMENT_METHOD to non-Visa rail.
   - **Don't**: Auto-retry Visa 3DS repeatedly — compound the timeout issue

2. **MASTERCARD_3DS_FAILED (PSP-MC-001)**: Mastercard 3D Secure authentication failed
   - **Root cause**: Customer entered wrong OTP, or Mastercard's Directory Server rejected
   - **Resolution**: RETRY_PAYMENT with fresh 3DS. If OTP was wrong, customer must retry with correct OTP.
   - **Don't**: Escalate for wrong OTP — customer must retry

3. **RUPAY_3DS_UNAVAILABLE (PSP-RUPAY-001)**: RuPay 3DS service unavailable
   - **Root cause**: NPCI's RuPay 3DS infrastructure under maintenance or outage
   - **Resolution**: WAIT_AND_RETRY after 30 minutes. Check NPCI status page.
   - **Don't**: Switch to Visa/MC — RuPay 3DS outages are usually brief

### Card Network Recovery Rules
- 3DS failures are usually transient — retry after 5-minute cooldown
- Maximum 2 retries before switching card network
- Card network 3DS outages usually affect all merchants simultaneously
- If customer's bank has recurring 3DS issues, suggest UPI Autopay as alternative
