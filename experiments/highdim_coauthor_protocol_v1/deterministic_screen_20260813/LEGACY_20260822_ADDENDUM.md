# Legacy status after the BatchNorm server-buffer correction

All BatchNorm-bearing image trajectories represented by this screen and its
expansions began before the corrected server-buffer policy was installed.
They are preserved for audit and debugging, but are scientifically ineligible:
do not resume them, select candidates from them, or mix them with corrected
trajectories.

The replacement workflow is the fresh
`deterministic_screen_post_bn_20260822` campaign. Its manifest is rebuilt from
the 108 unique intended configurations, but every run starts from
initialization with `server_buffer_policy=direct_client_aggregate`.
