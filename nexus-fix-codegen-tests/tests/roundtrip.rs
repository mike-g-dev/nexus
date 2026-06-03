use nexus_fix_codegen_tests::{venue_alpha, venue_beta};

#[test]
fn alpha_decodes_scalar_fields_and_enum() {
    let msg = b"11=ORD123\x0154=1\x0155=BTC-USD\x0138=10\x01";
    let m = venue_alpha::messages::NewOrderSingle::decode(msg);
    assert_eq!(m.cl_ord_id(), Some(&b"ORD123"[..]));
    assert_eq!(m.symbol(), Some(&b"BTC-USD"[..]));
    assert_eq!(m.order_qty(), Some(&b"10"[..]));
    assert_eq!(m.side_enum(), Some(venue_alpha::fields::Side::BUY));
}

#[test]
fn alpha_absent_field_is_none() {
    let msg = b"11=ORD123\x01";
    let m = venue_alpha::messages::NewOrderSingle::decode(msg);
    assert_eq!(m.symbol(), None);
    assert_eq!(m.side_enum(), None);
}

#[test]
fn alpha_decodes_data_field_with_embedded_soh() {
    let msg = b"11=A\x0195=3\x0196=a\x01b\x0155=X\x01";
    let m = venue_alpha::messages::NewOrderSingle::decode(msg);
    assert_eq!(m.raw_data_length(), Some(&b"3"[..]));
    assert_eq!(m.raw_data(), Some(&b"a\x01b"[..]));
    assert_eq!(m.symbol(), Some(&b"X"[..]));
}

#[test]
fn alpha_decodes_repeating_group() {
    let msg = b"11=A\x01453=2\x01448=PARTY1\x01452=1\x01448=PARTY2\x01452=2\x0155=X\x01";
    let m = venue_alpha::messages::NewOrderSingle::decode(msg);
    let parties: Vec<_> = m.no_party_i_ds().collect();
    assert_eq!(parties.len(), 2);
    assert_eq!(parties[0].party_id(), Some(&b"PARTY1"[..]));
    assert_eq!(parties[1].party_id(), Some(&b"PARTY2"[..]));
    assert_eq!(m.symbol(), Some(&b"X"[..]));
}

#[test]
fn alpha_decodes_nested_group() {
    let msg = b"11=A\x01453=1\x01448=P1\x01452=1\x01802=2\x01523=S1\x01803=1\x01523=S2\x01803=2\x0155=X\x01";
    let m = venue_alpha::messages::NewOrderSingle::decode(msg);
    let parties: Vec<_> = m.no_party_i_ds().collect();
    assert_eq!(parties.len(), 1);
    assert_eq!(parties[0].party_id(), Some(&b"P1"[..]));
    let subs: Vec<_> = parties[0].no_party_sub_i_ds().collect();
    assert_eq!(subs.len(), 2);
    assert_eq!(subs[0].party_sub_id(), Some(&b"S1"[..]));
    assert_eq!(subs[1].party_sub_id(), Some(&b"S2"[..]));
    assert_eq!(m.symbol(), Some(&b"X"[..]));
}

#[test]
fn alpha_decodes_execution_report() {
    let msg = b"37=ORD1\x0117=EX1\x01150=0\x0139=2\x0155=BTC-USD\x0154=1\x0132=5\x0131=100\x01";
    let m = venue_alpha::messages::ExecutionReport::decode(msg);
    assert_eq!(m.order_id(), Some(&b"ORD1"[..]));
    assert_eq!(m.exec_id(), Some(&b"EX1"[..]));
    assert_eq!(m.exec_type_enum(), Some(venue_alpha::fields::ExecType::NEW));
    assert_eq!(
        m.ord_status_enum(),
        Some(venue_alpha::fields::OrdStatus::FILLED)
    );
    assert_eq!(m.last_qty(), Some(&b"5"[..]));
    assert_eq!(m.last_px(), Some(&b"100"[..]));
}

#[test]
fn alpha_msgtype_dispatch() {
    use venue_alpha::MsgType;
    assert_eq!(MsgType::from_bytes(b"D"), Some(MsgType::NewOrderSingle));
    assert_eq!(MsgType::from_bytes(b"8"), Some(MsgType::ExecutionReport));
    assert_eq!(MsgType::from_bytes(b"0"), Some(MsgType::Heartbeat));
    assert_eq!(MsgType::ExecutionReport.as_bytes(), b"8");
    assert_eq!(MsgType::from_bytes(b"ZZ"), None);
}

#[test]
fn alpha_encodes_round_trip() {
    let mut buf = [0u8; 128];
    let n = venue_alpha::encoders::NewOrderSingleEncoder::new(&mut buf)
        .cl_ord_id(b"ORD1")
        .side_value(venue_alpha::fields::Side::SELL)
        .symbol(b"ETH-USD")
        .finish();
    let m = venue_alpha::messages::NewOrderSingle::decode(&buf[..n]);
    assert_eq!(m.cl_ord_id(), Some(&b"ORD1"[..]));
    assert_eq!(m.side_enum(), Some(venue_alpha::fields::Side::SELL));
    assert_eq!(m.symbol(), Some(&b"ETH-USD"[..]));
}

#[test]
fn alpha_encodes_data_field() {
    let mut buf = [0u8; 64];
    let n = venue_alpha::encoders::NewOrderSingleEncoder::new(&mut buf)
        .cl_ord_id(b"A")
        .raw_data(b"x\x01y")
        .finish();
    let m = venue_alpha::messages::NewOrderSingle::decode(&buf[..n]);
    assert_eq!(m.raw_data_length(), Some(&b"3"[..]));
    assert_eq!(m.raw_data(), Some(&b"x\x01y"[..]));
}

#[test]
fn beta_decodes_market_data_group() {
    let msg = b"55=EUR/USD\x01268=2\x01269=0\x01270=1.1050\x01271=1000000\x01269=1\x01270=1.1052\x01271=2000000\x01";
    let m = venue_beta::messages::MarketDataSnapshotFullRefresh::decode(msg);
    assert_eq!(m.symbol(), Some(&b"EUR/USD"[..]));
    let entries: Vec<_> = m.no_md_entries().collect();
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].md_entry_type_enum(),
        Some(venue_beta::fields::MDEntryType::BID)
    );
    assert_eq!(entries[0].md_entry_px(), Some(&b"1.1050"[..]));
    assert_eq!(
        entries[1].md_entry_type_enum(),
        Some(venue_beta::fields::MDEntryType::OFFER)
    );
    assert_eq!(entries[1].md_entry_size(), Some(&b"2000000"[..]));
}

#[test]
fn beta_msgtype_dispatch() {
    use venue_beta::MsgType;
    assert_eq!(
        MsgType::from_bytes(b"W"),
        Some(MsgType::MarketDataSnapshotFullRefresh)
    );
    assert_eq!(MsgType::from_bytes(b"A"), Some(MsgType::Logon));
    assert_eq!(MsgType::from_bytes(b"D"), None);
}

#[test]
fn modules_are_independent() {
    assert_eq!(venue_alpha::BEGIN_STRING, b"FIX.4.4");
    assert_eq!(venue_beta::BEGIN_STRING, b"FIX.4.2");
}
