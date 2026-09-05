# RBI Mandate & UPI Policies — Payment Recovery Reference

## RBI E-Mandate Framework (2021)

### Key Circular: RBI/2021-22/60 (AFA for Recurring Transactions)
- All recurring transactions above ₹5,000 require Additional Factor Authentication (AFA) for each renewal
- Banks must send pre-transaction notification 24 hours before debit
- Customer can opt-out of notification for transactions up to ₹15,000 (UPI Autopay limit)

### UPI Autopay Limits (NPCI Circular)
- **Maximum mandate amount**: ₹15,000 per transaction
- **Below ₹15,000**: Customer can set up automatic debits without per-transaction AFA
- **Above ₹15,000**: Each debit requires explicit customer authentication (OTP/PIN)
- **Mandate validity**: Maximum 5 years from creation date
- **Modification**: Customer can modify amount or frequency at any time through UPI app

### Mandate Lifecycle States
1. **ACTIVE**: Mandate is operational, debits can be initiated
2. **PAUSED**: Customer temporarily paused — cannot debit until reactivated
3. **CANCELLED**: Mandate terminated by customer — requires new mandate creation
4. **EXPIRED**: Mandate exceeded validity period — requires re-authorization
5. **REJECTED**: Mandate creation was rejected by customer's bank

### Recovery Protocol for Inactive Mandates
- **Mandate PAUSED**: Notify customer to reactivate. Do NOT retry debit.
- **Mandate CANCELLED**: Send re-authorization link. Customer must create new mandate.
- **Mandate EXPIRED**: Same as CANCELLED — new mandate required.
- **Recovery timeline**: 3 attempts over 7 days before escalating to human agent.

## RBI Dunning Guidelines

### Communication Frequency Limits (RBI/2022-23/56)
- **Maximum 3 reminder messages** per failed transaction
- **Minimum 24-hour gap** between consecutive reminders
- **Quiet hours**: No communications between 9:00 PM and 8:00 AM local time
- **Opt-out respect**: If customer opts out, block all further communications for that transaction
- **Language**: Communications must be in customer's preferred language

### Dunning Escalation Ladder
1. **Attempt 1** (T+0): Immediate retry (if transient error)
2. **Attempt 2** (T+24h): SMS/Email reminder with payment link
3. **Attempt 3** (T+48h): Push notification or WhatsApp with incentive (5% discount)
4. **Escalation** (T+72h): Human agent callback for high-value (>₹10,000) transactions

### Content Requirements
- Must include transaction reference number
- Must include amount and merchant name
- Must include link to retry payment
- Must NOT include sensitive details (card number, UPI PIN, etc.)
- Must include opt-out mechanism in every communication

## RBI Refund & Chargeback Rules

### Mandatory Refund Timeline (RBI/2019-20/177)
- **Failed transactions**: Refund within 5 business days of failure notification
- **Successful transactions (disputed)**: Refund within 8 weeks of complaint
- **Credit card chargebacks**: Merchant must respond within 30 days

### Auto-Refund Triggers
- Payment captured but order not fulfilled within 7 days
- Duplicate transaction detected (same amount, same customer, within 5 minutes)
- Gateway timeout after payment capture — order status unclear

## PCI-DSS Compliance in Recovery Communications

### Prohibited Data in Communications
- NEVER include full card number in any message (show only last 4: ****1234)
- NEVER include CVV or PIN in any communication
- NEVER include full UPI PIN in any message
- NEVER log full card numbers in audit trails

### Required Data Handling
- Payment IDs are safe to include (they are not sensitive)
- Order IDs are safe to include
- Error codes and descriptions are safe to include
- Customer email/name are safe but must respect privacy regulations

## Cross-Border Transaction Rules

### RBI LRS (Liberalised Remittance Scheme) Limits
- **Current account**: $10,000 per transaction
- **Capital account**: $250,000 per financial year
- **Recurring payments**: Require explicit customer authorization for each debit

### Recovery Considerations for International Cards
- 3DS authentication may fail due to cross-border latency
- Bank may block international transactions by default
- Customer must enable international transactions on their card
- Recovery protocol: RETRY with 3DS → if fails, UPDATE_PAYMENT_METHOD to domestic rail
