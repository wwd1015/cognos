# A team on one runs folder (v1.4)

COGNOS is run locally. A team that cannot deploy it as a service can still work on the same
runs, because a run is a directory: each person points their own workbench at a shared folder
(`cognos ui --runs-dir <share>`). Code, profiles, plugins and intent documents belong in git;
runs do not (they hold the data snapshot, the sealed holdout and fitted models).

Three things had to be true first.

## Decision

1. **State changes are exclusive across machines.** The run's lock was a thread lock. It is
   now also a lock file (`<runs>/_locks/<run_id>.lock`) created exclusively, which holds on
   network shares where advisory file locks do not. It guards the read-modify-write of
   `state.json` only, never a running stage. A waiter that sees the same holder for 30 seconds
   of its own clock takes over (no clock is compared between machines); a holder only removes
   its own file. The engine already re-read state from disk before every change, so a
   decision made from a stale page is refused by the ordinary rules (the gate is no longer
   awaiting).
2. **Decisions carry a name.** `identity.whoami()` is `COGNOS_USER`, else the login name. It
   is recorded on the run (`created_by`), on gate decisions and package seals (`by`), on
   answers (`answered_by`), and as `person@machine` on a step a process is running. It is a
   label, not authentication.
3. **A dead process can be taken over, by a person.** A laptop closed mid-stage leaves a step
   `running`. The engine cannot tell dead from slow, so it does nothing by itself:
   `cognos retry <step> --run <id> --stuck` resets it, and records who took it over from whom.

## Consequences

- Not access control. Seats are still chosen, not assigned; anyone who can write to the folder
  can act in any seat under any name. Binding seats to people needs real authentication.
- Use a network share, not a folder that syncs copies (OneDrive, Dropbox): a lock file that
  syncs late protects nothing, and conflicting copies of `state.json` appear.
- One person per run at a time is still the working rule. The lock prevents lost updates, not
  two people deciding different things in sequence.
- Deleting a run (`cognos delete-run`, the runs page) removes it for everyone.
