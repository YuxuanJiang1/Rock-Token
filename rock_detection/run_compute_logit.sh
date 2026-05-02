#!/bin/bash
python compute_logit_gradients.py --student onpolicy --samples 500 --hardware single_96gb --occurrences-file rock_token_occurrences_onpolicy_n500_unrestricted.pt
python analyze_gradient_alignment.py