"""The unchanged bank, with depth 3 explicitly selected as the main metric."""
from .evaluation_v3 import Evaluator as Previous


class Evaluator(Previous):
    def evaluate(self,params):
        result = super().evaluate(params)
        cells = result['by_suite_role_depth']
        def aggregate(suites):
            chosen = [cells[f'{s}:{r}:3'] for s in suites for r in (0,1)]
            counts = {k:sum(c[k] for c in chosen) for k in ('cases','wins','draws','losses')}
            n = counts['cases']
            return dict(counts,win_rate=counts['wins']/n,draw_rate=counts['draws']/n,
                loss_rate=counts['losses']/n,score_rate=(counts['wins']+.5*counts['draws'])/n)
        result.update(primary_depth=3,primary=aggregate(('empty','opening')),
            primary_by_suite={s:aggregate((s,)) for s in ('empty','opening')})
        return result
