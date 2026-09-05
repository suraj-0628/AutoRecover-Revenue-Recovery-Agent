# Merchant Dunning Best Practices

## Dunning Strategy by Transaction Amount

### Micro-Transactions (₹1 - ₹500)
- **Retry window**: 24 hours (2 attempts)
- **Communication**: SMS only (no email — cost > value)
- **Incentive**: None (cost of incentive exceeds transaction value)
- **Escalation**: Auto-write-off after 2 failed attempts (not worth human agent time)
- **Expected recovery**: 40-50%

### Small Transactions (₹500 - ₹5,000)
- **Retry window**: 48 hours (3 attempts)
- **Communication**: SMS + Email
- **Incentive**: 5% discount on retry within 24 hours
- **Escalation**: Auto-write-off after 3 attempts for amounts < ₹1,000
- **Expected recovery**: 55-65%

### Medium Transactions (₹5,000 - ₹50,000)
- **Retry window**: 72 hours (4 attempts)
- **Communication**: SMS + Email + Push notification
- **Incentive**: 5-10% discount tiered by urgency
- **Escalation**: Human agent callback for amounts > ₹25,000 after 3 attempts
- **Expected recovery**: 65-75%

### Large Transactions (₹50,000 - ₹5,00,000)
- **Retry window**: 7 days (5 attempts)
- **Communication**: All channels (SMS + Email + Push + WhatsApp + Phone call)
- **Incentive**: 10-15% discount + free shipping/upgrade
- **Escalation**: Human agent within 24 hours for all failed attempts
- **Expected recovery**: 70-80%

### High-Value Transactions (> ₹5,00,000)
- **Retry window**: 14 days (6 attempts)
- **Communication**: All channels + dedicated relationship manager
- **Incentive**: Custom negotiation (up to 20% discount)
- **Escalation**: Immediate human agent assignment
- **Expected recovery**: 75-85%

## Friction Thresholds

### Acceptable Friction Index
- **Target**: ≤ 0.3 (low friction)
- **Warning**: 0.3 - 0.5 (moderate friction — review dunning messages)
- **Critical**: > 0.5 (high friction — customer is being annoyed)
- **Action on critical**: Reduce communication frequency, simplify retry flow

### Communication Fatigue Rules
- **Maximum per 24 hours**: 2 communications (SMS + Email = 2)
- **Minimum gap**: 8 hours between communications
- **Maximum total per transaction**: 5 communications
- **Opt-out response time**: Must block within 1 hour of opt-out request

### Retry Attempt Spacing
- **Attempt 1**: Immediate (if transient error) or T+0
- **Attempt 2**: T+24 hours (with SMS reminder)
- **Attempt 3**: T+48 hours (with email + discount incentive)
- **Attempt 4**: T+72 hours (with push notification + higher discount)
- **Attempt 5**: T+168 hours (final attempt, human agent involved)

## Discount Tier Strategy

### Dynamic Discounting
| Days Since Failure | Discount | Communication Channel | Expected Lift |
|-------------------|----------|----------------------|---------------|
| 0-1 day | 0% | None (auto-retry) | +15% recovery |
| 1-2 days | 5% | SMS | +10% recovery |
| 2-3 days | 5% | Email | +8% recovery |
| 3-5 days | 10% | Push notification | +5% recovery |
| 5-7 days | 15% | WhatsApp | +3% recovery |
| 7+ days | 20% | Phone call | +2% recovery |

### Discount Cap Rules
- **Maximum total discount**: 20% of transaction value
- **Minimum transaction for discount**: ₹500 (below this, discount is not worthwhile)
- **Discount validity**: 48 hours from offer
- **One-time only**: Same customer cannot receive discount for same transaction twice

## Cohort-Based Dunning

### Customer Segmentation for Dunning
1. **First-time customers**: Aggressive retry (up to 5 attempts) — high lifetime value potential
2. **Repeat customers**: Moderate retry (3-4 attempts) — balance recovery vs. retention
3. **High-LTV customers**: Gentle retry (2-3 attempts) — prioritize relationship over single transaction
4. **Churned customers**: Final attempt (1 attempt) — don't waste resources on lost causes

### Salary-Day Optimization
- **IN salary cycles**: 1st and 15th of month
- **Optimal retry window**: 12:01 AM on salary date
- **Avoid**: Retrying during 10:00 AM - 2:00 PM on salary days (banks overloaded)
- **Expected recovery boost**: +20-30% when retrying at payday vs. random time

### Weekend vs. Weekday Dunning
- **Weekend failures**: Defer retry to Monday (banks process faster on weekdays)
- **Friday failures**: Retry Saturday morning (before weekend bank load)
- **Holiday failures**: Defer to next business day (no bank processing on holidays)

## Communication Templates

### SMS Template (160 chars max)
```
{merchant}: Payment of INR {amount} failed. Retry now: {link} - {discount}% off if you pay in 24h. Opt-out: STOP
```

### Email Template
```
Subject: Action needed: Your payment of INR {amount} to {merchant}

Hi {customer_name},

Your payment of INR {amount} to {merchant} couldn't be processed.

Reason: {failure_reason}

Click below to retry with {discount}% discount:
{payment_link}

Offer valid for 48 hours.

If you have questions, reply to this email.

- {merchant} Payments Team
```

### Push Notification Template
```
Your {merchant} payment (INR {amount}) needs attention. Tap to retry with {discount}% off: {deep_link}
```

## Compliance Checklist

- [ ] Communications respect quiet hours (9 PM - 8 AM)
- [ ] Maximum 2 communications per 24 hours
- [ ] Opt-out mechanism included in every message
- [ ] No sensitive data (card numbers, UPI PIN) in messages
- [ ] Transaction reference included in all communications
- [ ] Discount terms clearly stated (validity, one-time use)
- [ ] Customer data handled per PCI-DSS requirements
- [ ] Audit trail maintained for all dunning actions
