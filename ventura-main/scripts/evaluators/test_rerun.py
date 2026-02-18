import rerun as rr
import rerun.blueprint as rrb

rr.init("ventura_demo")  # <-- your app id
bp = rrb.Blueprint(...)

bp.save("ventura_demo", "ventura_bp.rbl")  # writes an .rbl file
